import json

from mlflow.pyfunc import Context, PythonModel

from kedro_framework.pipelines.ML.ts_model import TSModel


class PyMCModelWrapper(PythonModel):
    def load_context(self, context: Context) -> None:
        """
        Loads the pre-trained model from the given context.
        ---
        Params:
            context: The context object containing artifacts, including the model.
        """
        self.model = TSModel.load(context.artifacts['model'])

    def predict(self, context: Context, n_ahead: int) -> str:
        """
        Makes predictions using the model for the specified number of time steps ahead.
        ---
        Params:
            context: The context object used for prediction, including the necessary model data.
            n_ahead: The number of future time steps to predict.
        """
        preds_normalized = self.model.sample_posterior_predictive(
            n_ahead=n_ahead.iloc[:, 0].values.item(),
            extend_idata=False,
            combined=False
        )
        preds = preds_normalized * self.model.idata.attrs['y_std'] + self.model.idata.attrs['y_mean']
        preds = preds.rename_vars({"temperature_normalized_fut": "temperature_fut"})
        return json.dumps(preds.to_dataframe().reset_index().to_dict('records'))
