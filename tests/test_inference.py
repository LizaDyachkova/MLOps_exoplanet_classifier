import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestClassifier

from src import inference
from src.features import add_engineered_features
from src.inference import DEFAULT_CLASS_NAMES, ModelService

TRAIN = pd.DataFrame(
    {
        "koi_period": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "koi_prad": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "koi_depth": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        "koi_depth_err1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "koi_depth_err2": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    }
)
LABELS = [0, 1, 2, 0, 1, 2]


def make_service() -> ModelService:
    X = add_engineered_features(TRAIN)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, LABELS)
    return ModelService(
        model=model,
        class_names=DEFAULT_CLASS_NAMES,
        model_name="kepler_exoplanet_classifier",
        model_version="3",
    )


def test_load_model_service_reads_class_names_from_run():
    client_mock = MagicMock()
    client_mock.get_model_version_by_alias.return_value = MagicMock(
        version=3, run_id="run-1"
    )
    client_mock.get_run.return_value.data.params = {"class_names": "A,B,C"}
    fitted = make_service().model

    with (
        patch("src.inference.MlflowClient", return_value=client_mock),
        patch("src.inference.mlflow.sklearn.load_model", return_value=fitted),
    ):
        service = inference.load_model_service()

    assert service.class_names == ["A", "B", "C"]
    assert service.model_version == "3"


def test_load_model_service_falls_back_to_default_class_names():
    client_mock = MagicMock()
    client_mock.get_model_version_by_alias.return_value = MagicMock(
        version=1, run_id="run-1"
    )
    client_mock.get_run.return_value.data.params = {}

    with (
        patch("src.inference.MlflowClient", return_value=client_mock),
        patch(
            "src.inference.mlflow.sklearn.load_model", return_value=make_service().model
        ),
    ):
        service = inference.load_model_service()

    assert service.class_names == DEFAULT_CLASS_NAMES


def test_load_model_with_retries_gives_up():
    failing = MagicMock(side_effect=MlflowException("MLflow недоступен"))

    with (
        patch("src.inference.load_model_service", new=failing),
        patch("src.inference.asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        result = asyncio.run(inference.load_model_with_retries())

    assert result is None
    assert failing.call_count == inference.MODEL_LOAD_ATTEMPTS
    assert sleep.await_count == inference.MODEL_LOAD_ATTEMPTS - 1


def test_load_model_with_retries_succeeds_after_failure():
    service = make_service()
    flaky = MagicMock(side_effect=[OSError("нет связи"), service])

    with (
        patch("src.inference.load_model_service", new=flaky),
        patch("src.inference.asyncio.sleep", new=AsyncMock()),
    ):
        result = asyncio.run(inference.load_model_with_retries())

    assert result is service
    assert flaky.call_count == 2


def test_engineered_features_are_added():
    result = add_engineered_features(
        pd.DataFrame(
            {
                "koi_depth": [100.0],
                "koi_depth_err1": [2.0],
                "koi_depth_err2": [-2.0],
                "koi_prad": [2.0],
                "koi_prad_err1": [0.5],
                "koi_srad": [1.0],
                "koi_steff": [5778.0],
            }
        )
    )

    assert abs(result["depth_rel_err"][0] - 0.02) < 1e-9
    assert abs(result["prad_rel_err"][0] - 0.25) < 1e-9
    assert abs(result["star_luminosity_proxy"][0] - 1.0) < 1e-9


def test_predict_returns_classes_and_probabilities():
    result = make_service().predict(
        {"koi_period": 3.0, "koi_prad": 1.5, "koi_depth": 300.0}
    )

    assert result["prediction"] in DEFAULT_CLASS_NAMES
    assert set(result["probabilities"]) == set(DEFAULT_CLASS_NAMES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
    assert result["missing_features"] >= 0
