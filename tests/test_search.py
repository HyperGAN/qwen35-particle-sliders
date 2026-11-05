"""Run real CPU particle trials through the search's child boundary and cache checks."""
from types import SimpleNamespace
from subprocess import STDOUT
from pathlib import Path
import json

import pytest
import torch

from qwen35_sliders.data import write_json
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
    spec = dict(output=str(tmp_path/'search'),
        base=dict(data=str(data),device='cpu',seed=7,steps=2,batch=2,
                  checkpoint_every=1,calibration_rows=32),
        trials=[dict(lr=1e-5),dict(lr=2e-5)],final_steps=3)
    return spec,calls


def test_search_actual_trials_common_off_raw_sse_ranking_and_cache_reuse(search_fixture):
    from pathlib import Path
    spec,calls = search_fixture
    search_module.search(spec)
    assert len(calls)==3 and calls[-1]['steps']==3
    root = Path(spec['output'])
    results = json.loads((root/'search-results.json').read_text())
    selected = json.loads((root/'selected.json').read_text())
    best = min(results,key=lambda r:r['result']['student']['on_sse'])
    assert selected['trial']==best['index']
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
