import pandas as pd
import pytest

from house_price_ml.data import load_dataset, split_features_target


def test_load_dataset_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")


def test_split_features_target_rejects_missing_target() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column"):
        split_features_target(df, "target")
