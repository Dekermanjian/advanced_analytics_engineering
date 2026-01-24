"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import process_datasets, timeseries_plot


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_datasets,
                inputs=dict(partitioned_dataset="cities_temperatures"),
                outputs="cities_temperatures_processed",
                name="process_datasets",
            ),
            node(
                func=timeseries_plot,
                inputs=dict(processed_dataframe="cities_temperatures_processed"),
                outputs="cities_ts_plot",
                name="timeseries_plot",
            ),
        ]
    )
