# src/<package_name>/hooks.py
import os
import re
import tempfile
from typing import Any

import mlflow
from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
from mlflow.models.signature import ModelSignature

from kedro_framework.pipelines.ML.mlflow_model_wrapper import PyMCModelWrapper

cities = ["Belem", "Curitiba", "Fortaleza", "Goiania", "Macapa", "Manaus", "Recife", "Rio", "Salvador", "Sao_Luiz", "Sao_Paulo", "Vitoria"]

class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    def __init__(self):
        self.run_ids = {}
        conf_path = "./conf"
        conf_loader = OmegaConfigLoader(conf_source=conf_path)
        self.params = conf_loader.get('parameters')
        if self.params['mlflow_config']['use_mlflow']:
            # set the experiment and start an MLFlow run
            mlflow.set_experiment(experiment_name=self.params['mlflow_config']['experiment_name'])
            mlflow.start_run()
            # Save the run ID so that we can nest in the individual city models
            self.run_ids["parent_run_id"] = mlflow.active_run().info.run_id
            # We can log our model and sampler configs early
            mlflow.log_params(self.params['model_config'])
            mlflow.log_params(self.params['sampler_config'])
            # If you have tags set for the run set them in MLFlow
            if self.params['mlflow_config']['tags']:
                mlflow.set_tags(self.params['mlflow_config']['tags'])

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """
        Here we are going to pull outputs from specific nodes and log them with MLFlow
        ---
        Params:
            node: Attributes of the node that just ran
            outputs: Outputs of the node that just ran
            inputes: Inputes of the node that just ran
        """
        if node.name == "train":
            for city in cities:
                if self.params['mlflow_config']['use_mlflow']:
                    # Start a nested run
                    mlflow.start_run(run_name=city, nested=True, parent_run_id=self.run_ids['parent_run_id'])
                    # Store city specific run ids for later
                    self.run_ids[f"{city}_run_id"] = mlflow.active_run().info.run_id
                    # If you want to log artifacts log them here
                    if self.params['mlflow_config']['log_artifacts']:
                        local_path = f"./data/08_reporting/{city}_model_graph.png"
                        outputs[f"{city}_model"].model.to_graphviz(save=local_path, figsize=(12,8))
                        # log graph representation of model
                        mlflow.log_artifact(local_path=local_path, artifact_path="figures")
                        # Define the inputs the model expects when forecasting
                        input_schema = mlflow.types.Schema(
                            [
                                mlflow.types.ColSpec(name="n_ahead", type=mlflow.types.DataType.integer),
                            ]
                        )
                        # Log the model
                        with tempfile.TemporaryDirectory() as tmpdir:
                            file_path = os.path.join(tmpdir, f"{city}_model.nc")
                            outputs[f"{city}_model"].save(file_path)
                            mlflow.pyfunc.log_model(
                                artifact_path="model",
                                python_model=PyMCModelWrapper(),
                                artifacts={"model": file_path},
                                signature=ModelSignature(inputs=input_schema),
                                conda_env="./environment.yml"
                            )
                    # log divergences and inference data attributes
                    mlflow.log_param("divergences", outputs[f"{city}_model"].idata.sample_stats.diverging.sum().values.item())
                    mlflow.log_params(outputs[f"{city}_model"].idata.attrs)
                    mlflow.end_run()

        if node.name in [f"{city}_forecasts_evaluation" for city in cities]:
            city = re.search(r"(.*)(?=_forecasts_evaluation)", node.name).group(1)
            if self.params['mlflow_config']['use_mlflow']:
                # start up again our city specific runs to log metrics
                mlflow.start_run(run_id=self.run_ids[f"{city}_run_id"], run_name=city, nested=True, parent_run_id=self.run_ids['parent_run_id'])
                mlflow.log_metrics(outputs[f'{city}_evaluation'])
                mlflow.end_run()


    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        if mlflow.active_run():
            mlflow.end_run()
