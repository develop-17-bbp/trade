import pandas as pd
import numpy as np
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
