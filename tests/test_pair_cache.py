import json

import pytest
import torch

from qwen35_sliders.pair_cache import FrozenPairCache


def row():
    return dict(ids=[1,2], positive_ids=[3,4], neutral_text='question', positive_text='solution and question',
                history_ids=[], pair_provenance=dict(dataset_sha256='data', split='train', source_row_index=0,
                    teacher_generation_key='seed-and-answer-record', jev_key='correctness-record'))


def test_joint_pair_roundtrip_preserves_bits_and_linkage_without_recomputation(tmp_path):
    calls = []
    neutral = torch.tensor([[0., -0., 1.2345678]])
    positive = torch.tensor([[8.765432, 3., 2.]])
    def compute():
        calls.append(1); return neutral, positive
    first = FrozenPairCache(tmp_path, {'parent':'frozen','dtype':'bf16'})
    a,b = first.get_pair(row(),compute,'cpu')
    second = FrozenPairCache(tmp_path, {'parent':'frozen','dtype':'bf16'})
    x,y = second.get_pair(row(),compute,'cpu')
    assert len(calls)==1 and first.computed==1 and second.hits==1
    assert torch.equal(a.view(torch.uint8),x.view(torch.uint8))
    assert torch.equal(b.view(torch.uint8),y.view(torch.uint8))
    assert torch.equal(a,neutral) and torch.equal(b,positive)
    assert first.references==second.references
    packet = json.loads(next((tmp_path/'latent-pair').glob('*.json')).read_text())
    assert packet['recipe']['neutral_ids']==[1,2] and packet['recipe']['positive_ids']==[3,4]
    assert packet['recipe']['provenance']['teacher_generation_key']=='seed-and-answer-record'


def test_a_changed_side_model_or_accepted_example_cannot_reuse_another_pair(tmp_path):
    calls = []
    def compute():
        calls.append(1); return torch.zeros(1,3),torch.ones(1,3)
    cache = FrozenPairCache(tmp_path, {'parent':'base'})
    cache.get_pair(row(),compute,'cpu')
    changed = row();changed['ids']=[7,8];cache.get_pair(changed,compute,'cpu')
    changed = row();changed['positive_ids']=[9,10];cache.get_pair(changed,compute,'cpu')
    changed = row();changed['pair_provenance']['teacher_generation_key']='other-seed';cache.get_pair(changed,compute,'cpu')
    FrozenPairCache(tmp_path, {'parent':'other'}).get_pair(row(),compute,'cpu')
    assert len(calls)==5


def test_corrupt_pair_is_refused_instead_of_regenerated(tmp_path):
    calls = []
    def compute():
        calls.append(1);return torch.zeros(1,3),torch.ones(1,3)
    cache=FrozenPairCache(tmp_path,{'parent':'base'});cache.get_pair(row(),compute,'cpu')
    path=next((tmp_path/'latent-pair').glob('*.json'));packet=json.loads(path.read_text())
    packet['value']['neutral'],packet['value']['target']=packet['value']['target'],packet['value']['neutral']
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError,match='integrity'):cache.get_pair(row(),compute,'cpu')
    assert len(calls)==1


def test_invalid_pair_is_never_persisted(tmp_path):
    cache=FrozenPairCache(tmp_path,{'parent':'base'})
    with pytest.raises(ValueError,match='dimensions'):
        cache.get_pair(row(),lambda:(torch.zeros(1,3),torch.ones(1,4)),'cpu')
    with pytest.raises(ValueError,match='finite'):
        cache.get_pair(row(),lambda:(torch.zeros(1,3),torch.full((1,3),float('nan'))),'cpu')
    assert not list((tmp_path/'latent-pair').glob('*.json'))
