"""Run real CPU particle trials through the search's child boundary and cache checks."""
from types import SimpleNamespace
from subprocess import STDOUT
from pathlib import Path
import json

import pytest
import torch

from qwen35_sliders.data import digest, write_json
from qwen35_sliders import search as search_module
from qwen35_sliders import training
from test_host import TinyQwen


@pytest.fixture
def search_fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = tmp_path/'data.json'
    write_json(data, {key:[0,1,2,3] for key in ('train','dev','calibration')})
    ids = [[1,2,3],[2,3,4],[4,5,6],[7,8,9]]
    monkeypatch.setattr(training,'token_rows',lambda *args,**kwargs:
        [dict(ids=row,positive_ids=row+[10]) for row in ids])
    calls = []
    def child(args, **kwargs):
        config = json.loads(Path(args[-1]).read_text())
        calls.append(config)
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = TinyQwen()
        training.train_particle(config,runner=(SimpleNamespace(model=model,tokenizer=None),{'parent':'fixture-base'}))
    monkeypatch.setattr(search_module,'subprocess',SimpleNamespace(run=child, STDOUT=STDOUT))
    def evaluate(workspace, output, *, device, teacher, strength, offset, limit, seed, max_tokens):
        scores = [0, 1] if teacher is None else [1, int(teacher.parent.name == 'trial-00')]
        rows = [dict(example_index=i, sample=0, score=score, truncated=False)
                for i, score in enumerate(scores)]
        generations = output.with_suffix('.generations.jsonl')
        generations.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        write_json(output, dict(model_id='fixture', source_version_id='version', config_id='cfg',
            validation_sha256='manifest', runtime='CPU fixture', generations_sha256=digest(generations),
            artifact=dict(teacher_sha256=digest(teacher) if teacher else None, strength=strength,
                          deltas_sha256=None, norm_gain=1.0),
            protocol=dict(indices=[0, 1], selection_offset=offset, seed=seed,
                          generation=dict(max_tokens=max_tokens))))
    monkeypatch.setattr(search_module, 'evaluate_math', evaluate)
    spec = dict(output=str(tmp_path/'search'),
        development=dict(offset=0, limit=2, seed=1000, max_tokens=8192),
        base=dict(workspace='fixture',data=str(data),device='cpu',seed=7,steps=2,batch=2,
                  checkpoint_every=1,calibration_rows=32),
        trials=[dict(lr=1e-5),dict(lr=2e-5)],final_steps=3)
    return spec,calls


def test_search_ranks_finished_answers_and_reuses_cached_results(search_fixture):
    from pathlib import Path
    spec,calls = search_fixture
    search_module.search(spec)
    assert len(calls)==3 and calls[-1]['steps']==3
    root = Path(spec['output'])
    results = json.loads((root/'search-results.json').read_text())
    selected = json.loads((root/'selected.json').read_text())
    assert selected['trial'] == 0
    assert [r['intended_effect'] for r in results] == [0.5, 0.0]
    assert len({r['result']['student']['off_sse'] for r in results})==1
    for record in results:
        values=record['result']['student']
        assert values['on_sse']==pytest.approx(values['on_rms']**2*values['compared_values'])
        assert record['student_relative_sse']==pytest.approx(record['student_relative_rms']**2)
    search_module.search(spec)
    assert len(calls)==3
    contract = root/'trial-00/contract.json'
    before = contract.read_bytes()
    summary_before = (root/'search-results.json').read_bytes()
    changed = {**spec,'trials':[spec['trials'][0],dict(lr=3e-5)]}
    with pytest.raises(ValueError,match='Cached recipe differs'):
        search_module.search(changed)
    assert contract.read_bytes()==before and len(calls)==3
    assert (root/'search-results.json').read_bytes()==summary_before


def test_search_refuses_changed_source_data_and_incomplete_result(search_fixture):
    from pathlib import Path
    spec,calls = search_fixture
    search_module.search(spec)
    root = Path(spec['output'])
    result_path=root/'trial-00/result.json'
    result=json.loads(result_path.read_text())
    result['completed_steps']=1
    write_json(result_path,result)
    with pytest.raises(ValueError,match='training horizon'):
        search_module.search(spec)
    result['completed_steps']=2
    write_json(result_path,result)
    data=Path(spec['base']['data'])
    data.write_text(data.read_text()+'\n')
    with pytest.raises(ValueError,match='source/data identity'):
        search_module.search(spec)
    assert len(calls)==3


def test_search_rejects_a_different_off_baseline(search_fixture):
    from pathlib import Path
    spec,calls=search_fixture
    search_module.search(spec)
    path=Path(spec['output'])/'trial-01/result.json'
    result=json.loads(path.read_text())
    result['student']['off_sse']*=2
    write_json(path,result)
    with pytest.raises(ValueError,match='ON/OFF baselines'):
        search_module.search(spec)
    assert len(calls)==3


def test_cached_search_refuses_replaced_raw_weights(search_fixture):
    spec, calls = search_fixture
    search_module.search(spec)
    raw = Path(spec['output']) / 'trial-00' / 'particle.safetensors'
    replacement = Path(spec['output']) / 'trial-01' / 'particle.safetensors'
    raw.write_bytes(replacement.read_bytes())
    with pytest.raises(ValueError, match='Cached RAW artifact differs'):
        search_module.search(spec)
    assert len(calls) == 3


def test_failed_candidate_does_not_prevent_testing_other_candidates(search_fixture, monkeypatch):
    from particle_sliders.candidate_failure import CandidateFailure

    spec, calls = search_fixture
    child = search_module.subprocess.run

    def fail_first(args, **kwargs):
        config = json.loads(Path(args[-1]).read_text())
        if Path(config["output"]).name == "trial-00":
            raise CandidateFailure("Finished text failed")
        child(args, **kwargs)

    monkeypatch.setattr(search_module.subprocess, "run", fail_first)
    search_module.search(spec)
    root = Path(spec["output"])
    results = json.loads((root / "search-results.json").read_text())
    assert results[0]["status"] == "catastrophic_failure"
    assert not (root / "trial-00/development-on.json").exists()
    assert json.loads((root / "selected.json").read_text())["trial"] == 1
    assert [Path(config["output"]).name for config in calls] == ["trial-01", "final"]
