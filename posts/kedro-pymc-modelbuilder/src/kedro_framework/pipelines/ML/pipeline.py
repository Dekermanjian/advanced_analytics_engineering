"""
This is a boilerplate pipeline 'ML'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_test_split, train, forecast, forecast_plot

cities = ["Belem", "Curitiba", "Fortaleza", "Goiania", "Macapa", "Manaus", "Recife", "Rio", "Salvador", "Sao_Luiz", "Sao_Paulo", "Vitoria"]

def create_pipeline(**kwargs) -> Pipeline:
    split_train_nodes =  [
        node(
            func=train_test_split,
            inputs=dict(
                data = "cities_temperatures_processed",
                testing_window="params:testing_window"
            ),
            outputs=["training_dataset", "testing_dataset"],
            name="train_test_split",
            tags=["train"]
        ),
        node(
            func=train,
            inputs=dict(
                training_dataset="training_dataset",
                model_config="params:model_config",
                sampler_config="params:sampler_config"
            ),
            outputs=dict(
                Belem_model = "Belem_model",
                Curitiba_model = "Curitiba_model",
                Fortaleza_model = "Fortaleza_model",
                Goiania_model = "Goiania_model",
                Macapa_model = "Macapa_model",
                Manaus_model = "Manaus_model",
                Recife_model = "Recife_model",
                Rio_model = "Rio_model",
                Salvador_model = "Salvador_model",
                Sao_Luiz_model = "Sao_Luiz_model",
                Sao_Paulo_model = "Sao_Paulo_model",
                Vitoria_model = "Vitoria_model"
            ),
            name="train",
            tags=["train"]
        )
    ]

    forecast_training_nodes = [
        node(
            func=forecast,
            inputs=dict(
                model=f"{city}_model",
                n_ahead="params:n_ahead",
                ground_truth="testing_dataset"
            ),
            outputs={"forecasts": f"{city}_forecasts_evaluation", "evaluations": f"{city}_evaluation"},
            name=f"{city}_forecasts_evaluation",
            tags=["train"]
        )
        for city in cities
    ]

    forecast_nodes = [
        node(
            func=forecast,
            inputs=dict(
                model=f"{city}_model",
                n_ahead="params:n_ahead"
            ),
            outputs=f"{city}_forecasts",
            name=f"{city}_forecast",
            tags=["forecast"]
        )
        for city in cities
    ]

    plot_node = [
        node(
            func=forecast_plot,
            inputs=dict(
                training_dataset="training_dataset",
                testing_dataset="testing_dataset",
                model=f"{city}_model",
                forecast=f"{city}_forecasts_evaluation"
            ),
            outputs=f"{city}_plot",
            name=f"{city}_forecast_plot",
            tags=["train", "forecast"]
        )
        for city in cities
    ]

    return pipeline(
        split_train_nodes
        + forecast_training_nodes
        + forecast_nodes
        + plot_node
    )
