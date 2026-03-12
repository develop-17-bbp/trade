import pandas as pd
import numpy as np
from pathlib import Path
from src.scripts import train_lgbm


def test_build_dataset_shape():
    # create simple synthetic data frame
    n = 50
    df = pd.DataFrame({
        'timestamp': np.arange(n),
        'open': np.linspace(100, 150, n),
        'high': np.linspace(101, 151, n),
        'low': np.linspace(99, 149, n),
        'close': np.linspace(100, 150, n),
        'volume': np.ones(n),
    })
    features, labels = train_lgbm.build_dataset(df)
    assert len(features) == n
    assert len(labels) == n
    # features should be list of dicts with at least one key
    assert isinstance(features[0], dict)
    assert isinstance(labels[0], int)


def test_default_model_out_for_core_assets():
    assert Path(train_lgbm.default_model_out("BTC/USDT")).as_posix().endswith("models/lgbm_btc.txt")
    assert Path(train_lgbm.default_model_out("ETH/USDT")).as_posix().endswith("models/lgbm_eth.txt")


def test_load_ohlcv_normalizes_common_column_names():
    path = Path("logs/test_train_lgbm_load_ohlcv.csv")
    pd.DataFrame({
        "Time": np.arange(3),
        "Open": [1.0, 2.0, 3.0],
        "High": [2.0, 3.0, 4.0],
        "Low": [0.5, 1.5, 2.5],
        "Close": [1.5, 2.5, 3.5],
        "Volume": [10.0, 11.0, 12.0],
    }).to_csv(path, index=False)

    try:
        df = train_lgbm.load_ohlcv(str(path))
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    finally:
        if path.exists():
            path.unlink()
