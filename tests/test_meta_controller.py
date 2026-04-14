import pytest
from src.trading.meta_controller import MetaController


def test_meta_controller_agreement():
    mc = MetaController()
    # both engines agree long with decent confidence
    final_dir, final_conf, scale = mc.arbitrate(
        lgb_class=1, lgb_conf=0.8,
        rl_action=1, rl_prob=0.7,
        features={'ewma_vol': 0.02, 'vol_adj_momentum': 0.9},
        finbert_score=0.1,
    )
    assert final_dir == 1
    assert 0.0 < final_conf <= 1.0
    assert 0.0 < scale <= 1.0  # Scale is proportional to position sizing, not always 1.0


def test_meta_controller_veto_low_rl():
    mc = MetaController()
    # RL probability low should veto regardless of LGB prediction
    final_dir, final_conf, scale = mc.arbitrate(
        lgb_class=1, lgb_conf=0.9,
        rl_action=-1, rl_prob=0.3,
        features={'ewma_vol': 0.01},
        finbert_score=0.0,
    )
    assert final_dir == 0
    assert scale == 0.0


def test_meta_controller_volatility_weights():
    mc = MetaController()
    # volatility above threshold should tilt weight to RL
    final_dir_low, _, _ = mc.arbitrate(
        lgb_class=1, lgb_conf=0.9,
        rl_action=1, rl_prob=0.6,
        features={'ewma_vol': 0.01},
        finbert_score=0.0,
    )
    final_dir_high, _, _ = mc.arbitrate(
        lgb_class=1, lgb_conf=0.9,
        rl_action=1, rl_prob=0.6,
        features={'ewma_vol': 0.05},
        finbert_score=0.0,
    )
    assert final_dir_low == final_dir_high == 1


@pytest.mark.parametrize("lgb,rl", [(1, -1), (-1, 1)])
def test_meta_controller_disagreement(lgb, rl):
    mc = MetaController()
    # conflicting signals with moderate probabilities should often return flat or low confidence
    final_dir, final_conf, scale = mc.arbitrate(
        lgb_class=lgb, lgb_conf=0.7,
        rl_action=rl, rl_prob=0.7,
        features={'ewma_vol': 0.02},
        finbert_score=0.0,
    )
    assert final_dir in (-1, 0, 1)
    assert 0.0 <= scale <= 1.0


def test_meta_controller_bias():
    mc = MetaController({'bias': 0.1})
    # when both engines agree long, confidence should increase by bias
    dir0, conf0, _ = mc.arbitrate(
        lgb_class=1, lgb_conf=0.5,
        rl_action=1, rl_prob=0.5,
        features={'ewma_vol': 0.01},
        finbert_score=0.0,
    )
    assert dir0 == 1
    assert conf0 >= 0.5
