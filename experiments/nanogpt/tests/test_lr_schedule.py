import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lr_schedule import scheduled_lr  # noqa: E402


def test_wsd_warmup_stable_decay():
    lr = lambda s: scheduled_lr(s, 100, 10, 20, 1., .1)
    assert lr(0) == pytest.approx(.1)
    assert lr(9) == lr(10) == lr(79) == lr(80) == 1.
    assert lr(90) == pytest.approx(.55)
    assert lr(99) == pytest.approx(.145)
    assert lr(100) == lr(101) == pytest.approx(.1)
    assert all(lr(i) >= lr(i + 1) for i in range(10, 110))


def test_wsd_without_decay_or_warmup():
    assert all(scheduled_lr(s, 50, 0, 0, 2.) == 2. for s in range(60))
    assert scheduled_lr(0, 10, 0, 10, 1.) == 1. and scheduled_lr(5, 10, 0, 10, 1.) == .5


def test_cosine():
    lr = lambda s: scheduled_lr(s, 100, 10, 0, 1., .1, 'cosine')
    assert lr(0) == pytest.approx(.1)
    assert lr(9) == lr(10) == 1.
    assert lr(55) == pytest.approx(.55)
    assert lr(100) == lr(101) == pytest.approx(.1)
    assert all(lr(i) >= lr(i + 1) for i in range(10, 100))
    assert scheduled_lr(10, 100, 10, 50, 1., 0., 'cosine') == 1.   # decay_steps is ignored


def test_invalid_arguments():
    with pytest.raises(ValueError):
        scheduled_lr(0, 100, 10, 0, 1., schedule='linear')
    with pytest.raises(ValueError):
        scheduled_lr(0, 100, 60, 50, 1.)
    with pytest.raises(ValueError):
        scheduled_lr(-1, 100, 10, 10, 1.)
    with pytest.raises(ValueError):
        scheduled_lr(0, 0, 0, 0, 1.)
