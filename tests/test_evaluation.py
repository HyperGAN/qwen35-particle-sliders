"""CPU tests for candidate cleanup and matched evaluation identities."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from qwen35_sliders.candidate import applied_teacher
from qwen35_sliders.data import digest
from qwen35_sliders.evaluation import document_hash
from qwen35_sliders.host import Slider, hidden
from qwen35_sliders.upstream import load
from test_host import TinyQwen


@pytest.fixture
def teacher(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = TinyQwen()
    slider = Slider(model, load())
    try:
        with torch.no_grad():
            for branch in slider.branches:
                branch.up.weight.fill_(.1)
        path = tmp_path/'teacher.safetensors'
        save_file({k:v.detach().contiguous() for k,v in slider.state_dict().items()}, path)
    finally:
        slider.close()
    return model, path


def test_teacher_exact_off_and_hooks_restored_after_evaluation_failure(teacher):
    model, path = teacher
    before = hidden(model, [1,2,3]).detach()
    flags = [p.requires_grad for p in model.parameters()]
    with applied_teacher(model, path, 0) as identity:
        assert identity['particlegan_version'] == '0.9.5'
        assert torch.equal(hidden(model, [1,2,3]), before)
    with pytest.raises(RuntimeError, match='evaluation failure'):
        with applied_teacher(model, path, 1):
            assert not torch.equal(hidden(model, [1,2,3]), before)
            raise RuntimeError('evaluation failure')
    assert torch.equal(hidden(model, [1,2,3]), before)
    assert [p.requires_grad for p in model.parameters()] == flags
    invalid = path.with_name('invalid.safetensors')
    save_file({'wrong':torch.ones(1)}, invalid)
    with pytest.raises(RuntimeError):
        with applied_teacher(model, invalid, 1):
            pytest.fail('invalid teacher entered evaluation')
    assert torch.equal(hidden(model, [1,2,3]), before)
    assert [p.requires_grad for p in model.parameters()] == flags


@pytest.fixture
def public_math(tmp_path, monkeypatch, teacher):
    datasets = pytest.importorskip('datasets')
    import delta_cli.local as local
    import delta_cli.scoring as scoring
    import qwen35_sliders.training as training
    model, path = teacher
    docs = [dict(question=f'Question {i}', answer='worked answer #### 42') for i in range(12)]
    selected = [9,2,7,4,11,5]
    manifest = dict(text={'gsm8k':{'selection':{'gsm8k':dict(indices=selected,
        doc_sha256=[document_hash(docs[i]) for i in selected], population=12)}}},
        dataset_revisions={'openai/gsm8k':'fixture-revision'})
    (tmp_path/'validation.json').write_text(json.dumps(manifest))
    config = dict(id='fixture-config', validation_sha256=digest(tmp_path/'validation.json'),
                  generation={'temperature':1., 'top_p':.95})
    monkeypatch.setattr(local, 'read_workspace', lambda workspace:
        (tmp_path, {'model_id':'fixture-model'}, config, {'id':'fixture-parent'}))
    monkeypatch.setattr(datasets, 'load_dataset', lambda *args, **kwargs:docs)
    monkeypatch.setattr(scoring, 'score_answer', lambda gold,text:float(text=='42'))
    calls = []
    def generate(prompt, generation, seed):
        calls.append(dict(prompt=prompt, generation=generation, seed=seed,
                          hidden=hidden(model,[1,2,3]).detach().clone()))
        return dict(text='42', tokens=2, truncated=False, generation_seconds=0.)
    runner = SimpleNamespace(model=model, generate=generate)
    monkeypatch.setattr(training, 'load_host', lambda *args:(runner, {'runtime':{'device':'cpu'}}))
    return tmp_path, path, calls, model


def test_math_offsets_and_seeds_bind_matched_comparisons(public_math):
    from delta_cli.compare import compare_reports
    from qwen35_sliders.math_evaluation import evaluate_math
    root, teacher, calls, model = public_math
    original = hidden(model,[1,2,3]).detach()
    settings = dict(teacher=teacher, limit=2, offset=2, seed=2000, device='cpu')
    off = evaluate_math(root, root/'off.json', strength=0, **settings)
    on = evaluate_math(root, root/'on.json', strength=1, **settings)
    assert off['validation_sha256'] == on['validation_sha256']
    assert off['protocol']['indices'] == [7,4]
    assert off['protocol']['selection_offset'] == 2 and off['protocol']['seed'] == 2000
    assert [c['prompt'] for c in calls] == ['Question 7','Question 4']*2
    assert all(c['seed'] == 2000 for c in calls)
    assert all('worked answer' not in c['prompt'] for c in calls)
    assert torch.equal(calls[0]['hidden'], original)
    assert not torch.equal(calls[2]['hidden'], original)
    assert torch.equal(hidden(model,[1,2,3]), original)
    assert compare_reports(root/'off.json',root/'on.json',resamples=50)['generation_differs'] is False
    evaluate_math(root, root/'different-seed.json', strength=1, **{**settings,'seed':2001})
    with pytest.raises(ValueError, match='different evaluation configuration'):
        compare_reports(root/'off.json',root/'different-seed.json',resamples=50)
    evaluate_math(root, root/'different-slice.json', strength=1, **{**settings,'offset':0})
    with pytest.raises(ValueError, match='different evaluation configuration'):
        compare_reports(root/'off.json',root/'different-slice.json',resamples=50)


def test_invalid_selection_refused_before_model_load(public_math, monkeypatch):
    from qwen35_sliders.math_evaluation import evaluate_math
    import qwen35_sliders.training as training
    root, teacher, _, _ = public_math
    monkeypatch.setattr(training, 'load_host', lambda *args:pytest.fail('model allocated'))
    with pytest.raises(ValueError, match='exceeds'):
        evaluate_math(root, root/'bad.json', limit=4,offset=3,device='cpu')


def test_likelihood_teacher_is_active_for_the_actual_evaluator(tmp_path, monkeypatch, teacher):
    import sys
    pytest.importorskip('lm_eval')
    from lm_eval import evaluator
    import delta_cli.local as local
    import qwen35_sliders.training as training
    import qwen35_sliders.evaluation as module
    model, path = teacher
    original = hidden(model,[1,2,3]).detach()
    (tmp_path/'validation.json').write_text(json.dumps({'text':{'arc_easy':{'selection':{'arc_easy':{}}}}}))
    config = dict(id='fixture-config',validation_sha256=digest(tmp_path/'validation.json'))
    monkeypatch.setattr(local,'read_workspace',lambda workspace:
        (tmp_path,{'model_id':'fixture-model'},config,{'id':'fixture-parent'}))
    monkeypatch.setattr(training,'load_host',lambda *args:
        (SimpleNamespace(model=model,tokenizer=None),{'runtime':{'device':'cpu'}}))
    monkeypatch.setattr(module,'prepare_tasks',lambda *args:
        (['fixture-task'],{'arc_easy':[0]}, {'arc_easy':{'indices':[0]}}))
    monkeypatch.setitem(sys.modules,'lm_eval.models.huggingface',
        SimpleNamespace(HFLM=lambda **kwargs:kwargs['pretrained']))
    observed = []
    def evaluate(**kwargs):
        assert kwargs['model'] is model
        observed.append(hidden(model,[1,2,3]).detach().clone())
        return {'samples':{'arc_easy':[{'doc_id':0,'acc':1.}]}}
    monkeypatch.setattr(evaluator,'simple_evaluate',evaluate)
    off = module.evaluate(tmp_path,tmp_path/'off.json',teacher=path,strength=0,
                          names=['arc_easy'],device='cpu')
    on = module.evaluate(tmp_path,tmp_path/'on.json',teacher=path,strength=1,
                         names=['arc_easy'],device='cpu')
    assert torch.equal(observed[0],original)
    assert not torch.equal(observed[1],original)
    assert torch.equal(hidden(model,[1,2,3]),original)
    assert off['validation_sha256'] == on['validation_sha256']
    assert on['artifact']['teacher_engine']['particlegan_version'] == '0.9.5'
