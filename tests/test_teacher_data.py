import copy
from concurrent.futures import ThreadPoolExecutor
import json
import time

import httpx
import pytest

from qwen35_sliders.teacher_data import Cache, ENDPOINT, Jev, QUESTIONS, PROMPT, fingerprint, merge, sample_positive, validate_splits
from qwen35_sliders.data import digest


ROW = dict(prompt="There are 8 red and 5 blue marbles. How many total?", reasoning="8+5=13.",
           answer="13", source="openai/gsm8k:train")


class Client:
    def __init__(self, status=200):
        self.calls = []
        self.status = status

    def post(self, endpoint, headers, json):
        assert endpoint == ENDPOINT
        self.calls.append(json)
        bad = "8+5=12" in json["state"]["candidate"]
        return httpx.Response(self.status, json=dict(model=json["model"], answers=dict(
            correct=dict(type="noul", noul=.1 if bad else .99),
            quality=dict(type="score", score=.5 if bad else 3.9, confidence=.9))))


def sampler(tmp_path, client=None, questions=None):
    cache = Cache(tmp_path)
    client = client or Client()
    judge = Jev(cache, key="test-key", client=client, questions=questions)
    return cache, judge, client


def test_rejection_sampling_resumes_without_generation_or_api_calls(tmp_path):
    cache, judge, client = sampler(tmp_path)
    calls = []
    answers = iter(["8+5=12.\n####12", "8+5=12.\n####13", "8+5=13. There are 13 marbles.\n####13"])
    def generate(prompt, generation, seed):
        assert 0 <= seed < 2**32  # Transformers seeds NumPy as well as Torch.
        calls.append(seed)
        return dict(text=next(answers), truncated=False)
    args = (ROW, generate, dict(parent="frozen"), dict(max_tokens=256), cache, judge)
    first = sample_positive(*args)
    assert first["accepted"]["attempt"] == 2
    assert len(calls) == 3 and len(set(calls)) == 3
    assert len(client.calls) == 2  # Wrong final number is rejected in code.
    second = sample_positive(*args)
    assert second["accepted"]["generation_key"] == first["accepted"]["generation_key"]
    assert len(calls) == 3 and len(client.calls) == 2


def test_changed_rubric_reranks_saved_answers_without_regeneration(tmp_path):
    cache, judge, client = sampler(tmp_path)
    calls = []
    def generate(*args):
        calls.append(1)
        return dict(text="8+5=13.\n####13", truncated=False)
    args = (ROW, generate, {"parent":"frozen"}, {}, cache)
    sample_positive(*args, judge)
    questions = copy.deepcopy(QUESTIONS)
    questions["quality"]["instructions"] += " Prefer a direct explanation."
    changed = Jev(cache, key="test", client=client, questions=questions)
    result = sample_positive(*args, changed)
    assert result["accepted"] and len(calls) == 1 and len(client.calls) == 2


def test_changed_teacher_or_seed_gets_a_new_generation_identity(tmp_path):
    cache, judge, _ = sampler(tmp_path)
    calls = []
    def generate(*args):
        calls.append(1)
        return dict(text="8+5=13.\n####13", truncated=False)
    for parent, seed in [("a", 7), ("b", 7), ("b", 8)]:
        sample_positive(ROW, generate, {"parent":parent}, {}, cache, judge, seed=seed)
    assert len(calls) == 3


def test_provider_failure_preserves_generated_answers_and_never_caches_error(tmp_path):
    cache, judge, _ = sampler(tmp_path, Client(401))
    calls = []
    def generate(*args):
        calls.append(1)
        return dict(text="8+5=13.\n####13", truncated=False)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        sample_positive(ROW, generate, {}, {}, cache, judge)
    assert len(list((tmp_path/"generation").glob("*.json"))) == 1
    assert not list((tmp_path/"jev").glob("*.json"))
    restored = Jev(cache, key="test", client=Client())
    assert sample_positive(ROW, generate, {}, {}, cache, restored)["accepted"]
    assert len(calls) == 1


def test_exhausted_budget_stays_pending_and_can_extend_without_repeating_attempts(tmp_path):
    cache, judge, _ = sampler(tmp_path)
    calls = []
    def generate(*args):
        calls.append(1)
        return dict(text="8+5=12.\n####12" if len(calls)<=2 else "8+5=13.\n####13", truncated=False)
    args = (ROW, generate, {}, {}, cache, judge)
    assert sample_positive(*args, max_attempts=2)["pending"]
    resumed = sample_positive(*args, max_attempts=3)
    assert resumed["accepted"]["attempt"] == 2 and len(calls) == 3


def test_truncation_and_token_budget_cannot_be_overridden_by_judge(tmp_path):
    cache, judge, client = sampler(tmp_path)
    def generate(*args):return dict(text="8+5=13.\n####13", truncated=True)
    assert sample_positive(ROW, generate, {}, {}, cache, judge, max_attempts=1)["pending"]
    assert not client.calls
    def complete(*args):return dict(text="8+5=13.\n####13", truncated=False)
    assert sample_positive(ROW, complete, {"parent":"new"}, {}, cache, judge, max_attempts=1,
                           fits=lambda _:False)["pending"]
    assert not client.calls


def test_cache_detects_tampering_and_serializes_duplicate_workers(tmp_path):
    cache = Cache(tmp_path)
    calls = []
    def compute():
        calls.append(1); time.sleep(.01); return {"text":"answer"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _:cache.once("generation", {"seed":7}, compute), range(4)))
    assert len(calls) == 1 and sum(not r[1] for r in results) == 1
    path = cache.path("generation", {"seed":7})
    packet = json.loads(path.read_text()); packet["value"]["text"] = "changed"; path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="integrity"):
        cache.get("generation", {"seed":7})


def test_evaluation_sources_and_split_overlap_are_refused():
    data = dict(train=[ROW], dev=[dict(ROW, prompt="dev")], calibration=[dict(ROW, prompt="cal")])
    validate_splits(data)
    data["dev"][0]["source"] = "openai/gsm8k:test"
    with pytest.raises(ValueError, match="never evaluation"):
        validate_splits(data)
    data["dev"] = [ROW]
    with pytest.raises(ValueError, match="disjoint"):
        validate_splits(data)


def test_merge_preserves_heldout_rows_and_rejects_selection_tampering(tmp_path):
    source = tmp_path/'source.json'
    data = dict(train=[dict(ROW, prompt=ROW['prompt']+str(i)) for i in range(3)],
                dev=[dict(ROW, prompt='heldout-dev')], calibration=[dict(ROW, prompt='heldout-cal')])
    source.write_text(json.dumps(data))
    cache, judge, _ = sampler(tmp_path/'cache')
    output = tmp_path/'curated.json'
    common = dict(source_data_sha256=digest(source), teacher={'parent':'frozen'}, generation={},
        generator_prompt_sha256=fingerprint(PROMPT), jev_model=judge.model, jev_endpoint=ENDPOINT,
        questions=QUESTIONS, policy=dict(seed=7100,min_correctness=.9,min_quality=3.,training_max_tokens=384))
    for shard in range(2):
        selected = []
        for i in range(shard,3,2):
            row = data['train'][i]
            result = sample_positive(row,lambda *args:dict(text='8+5=13.\n####13',truncated=False),
                                     common['teacher'],{},cache,judge)
            best = result['accepted']
            selected.append(dict(index=i,row=dict(row,reasoning=best['reasoning']),attempt=best['attempt'],
                generation_key=best['generation_key'],jev=best['jev'],attempts=result['attempts']))
        output.with_suffix(f'.part-{shard:02d}-of-02.json').write_text(json.dumps(
            dict(contract=dict(common,shard=shard,shards=2),selected=selected,pending=[])))
    config = dict(data=str(source),output=str(output),cache=str(tmp_path/'cache'))
    result = merge(config,shards=2)
    assert result['dev']==data['dev'] and result['calibration']==data['calibration']
    assert len(result['train'])==3
    assert merge(config,shards=2)==result  # Identical output is reused.
    output.unlink()
    part = output.with_suffix('.part-00-of-02.json')
    packet = json.loads(part.read_text());packet['selected'][0]['row']['reasoning']='Forged answer';part.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match='differs from the cached'):
        merge(config,shards=2)


def test_retry_schedule_reuses_accepted_legacy_answer_and_checks_actual_recipe(tmp_path):
    from qwen35_sliders.teacher_data import generation_recipe, verify_selection
    cache, judge, _ = sampler(tmp_path)
    calls = []
    def generate(prompt, settings, seed):
        calls.append(dict(settings))
        return dict(text="8+5=13. There are 13 marbles.\n####13", truncated=False)
    settings = dict(max_tokens=256, presence_penalty=1.5)
    original = sample_positive(ROW, generate, {"parent":"frozen"}, settings, cache, judge)
    retried = sample_positive(ROW, generate, {"parent":"frozen"}, settings, cache, judge,
        max_attempts=128, retry_generation={"presence_penalty":0., "temperature":.6})
    assert len(calls) == 1
    assert original["accepted"]["generation_key"] == retried["accepted"]["generation_key"]
    best = retried["accepted"]
    entry = dict(best, row=dict(ROW, reasoning=best["reasoning"]))
    contract = dict(teacher={"parent":"frozen"}, generation=settings,
        policy=dict(seed=7100,min_correctness=.9,min_quality=3.),
        retry_generation={"presence_penalty":0., "temperature":.6}, retry_after=64,
        jev_model=judge.model, questions=QUESTIONS)
    verify_selection(entry, ROW, contract, cache)
    assert generation_recipe(ROW, {"parent":"frozen"}, settings,7100,63)["generation"] == settings


def test_retry_schedule_generates_only_new_setting_and_verifies_selected_record(tmp_path):
    from qwen35_sliders.teacher_data import verify_selection
    cache, judge, client = sampler(tmp_path)
    calls = []
    def generate(prompt, settings, seed):
        calls.append(dict(settings))
        good = settings.get("presence_penalty") == 0.
        return dict(text="8+5=13.\n####13" if good else "8+5=12.\n####12", truncated=False)
    settings = dict(max_tokens=256, presence_penalty=1.5)
    old = sample_positive(ROW,generate,{"parent":"frozen"},settings,cache,judge,max_attempts=2)
    assert old["pending"] and len(calls)==2
    retry = {"presence_penalty":0., "temperature":.6}
    result = sample_positive(ROW,generate,{"parent":"frozen"},settings,cache,judge,
        max_attempts=128,retry_generation=retry)
    best=result["accepted"]
    assert best["attempt"]==64 and len(calls)==3 and calls[-1]==dict(settings,**retry)
    contract=dict(teacher={"parent":"frozen"},generation=settings,
        policy=dict(seed=7100,min_correctness=.9,min_quality=3.),retry_generation=retry,retry_after=64,
        jev_model=judge.model,questions=QUESTIONS)
    verify_selection(dict(best,row=dict(ROW,reasoning=best["reasoning"])),ROW,contract,cache)
    contract["retry_generation"]={"presence_penalty":.5}
    with pytest.raises(ValueError,match="generation identity differs"):
        verify_selection(dict(best,row=dict(ROW,reasoning=best["reasoning"])),ROW,contract,cache)
    with pytest.raises(ValueError,match="1..128"):
        sample_positive(ROW,generate,{},settings,cache,judge,max_attempts=129)
