"""Project pipelines."""

from kedro.pipeline import Pipeline

from kedro_framework.pipelines.data_processing import create_pipeline as dp
from kedro_framework.pipelines.ML import create_pipeline as ml


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    dp_pipeline = dp()
    ml_pipeline = ml()
    training_pipeline = ml_pipeline.only_nodes_with_tags("train")
    forecast_pipeline = ml_pipeline.only_nodes_with_tags("forecast")

    return {
        "data_processing": dp_pipeline,
        "train": training_pipeline,
        "forecast": forecast_pipeline,
        "__default__": dp_pipeline + training_pipeline
    }
