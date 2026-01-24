import warnings
from abc import abstractmethod

import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import xarray as xr
from pymc_extras.model_builder import ModelBuilder


class TSModel(ModelBuilder):
    _model_type = "TimeSeries"
    version = "0.1"

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        city: str = None
    ):
        """
        Initializes model configuration and sampler configuration for the model

        Parameters
        ----------
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Class-default defined by the user default_sampler_config method.
        city: The Brazilian city we are modeling monthly average temperatures of
        """
        sampler_config = (
            self.get_default_sampler_config() if sampler_config is None else sampler_config
        )
        self.sampler_config = sampler_config
        model_config = self.get_default_model_config() if model_config is None else model_config

        self.model_config = model_config  # parameters for priors etc.
        self.model = None  # Set by build_model
        self.idata: az.InferenceData | None = None  # idata is generated during fitting
        self.is_fitted_ = False
        self.city = city

    def _generate_and_preprocess_model_data(self, X: pl.DataFrame, y: pl.Series, normalize: bool = False):
        """
        Last mile data processing of inputs expected by the model
        ---
        Params:
            X: The matrix of predictor variables expected by our model
            y: The target variable
            normalize: Whether to Z normalize the variables
        """
        self.train_time_range = np.arange(X.shape[0])
        self.n_modes = 10
        periods = np.array(self.train_time_range) / (12)
        self.train_fourier_series = pl.DataFrame(
            {
                f"{func}_order_{order}": getattr(np, func)(
                    2 * np.pi * periods * order
                )
                for order in range(1, self.n_modes + 1)
                for func in ("sin", "cos")
            }
        )
        if normalize:
            self.y_mean = np.nanmean(y)
            self.y_std = np.nanstd(y)
            self.y_normalized = (y - self.y_mean) / self.y_std
        else:
            self.y_normalized = y

    def _data_setter(self, n_ahead: int):
        """
        Generates required data for producing forecasts
        ---
        Params:
            n_ahead: How many periods (months) to forecast future temperatures
        """
        self.start = self.train_time_range[-1]
        self.end = self.start + n_ahead

        new_periods = np.arange(self.start, self.end, 1) / (12)
        self.test_fourier_series = pl.DataFrame(
            {
                f"{func}_order_{order}": getattr(np, func)(
                    2 * np.pi * new_periods * order
                )
                for order in range(1, self.n_modes + 1)
                for func in ("sin", "cos")
            }
        )

    def build_model(self, X: pl.DataFrame, y: pl.Series, normalize_target: bool = False, **kwargs):
        """
        Defines the PyMC model structure
        ---
        Params:
            X: Dataframe of features
            y: Array of target values
        """

        self._generate_and_preprocess_model_data(X, y, normalize=normalize_target)

        with pm.Model() as self.model:
            self.model.add_coord("obs_id", self.train_time_range)
            self.model.add_coord(
                "fourier_features",
                np.arange(len(self.train_fourier_series.to_numpy().T)),
            )

            t = pm.Data("time_range", self.train_time_range, dims="obs_id")
            fourier_terms = pm.Data(
                "fourier_terms", self.train_fourier_series.to_numpy().T
            )

            error = pm.HalfNormal("error", self.model_config['error'])

            # Trend component
            amplitude_trend = pm.HalfNormal("amplitude_trend", self.model_config['amplitude_trend'])
            ls_trend = pm.Gamma("ls_trend", alpha=self.model_config['ls_trend']['alpha'], beta=self.model_config['ls_trend']['beta'])
            cov_trend = amplitude_trend * pm.gp.cov.ExpQuad(1, ls_trend)

            gp_trend = pm.gp.HSGP(
                m=[10],
                c=5.,
                cov_func=cov_trend
            )
            trend = gp_trend.prior("trend", X=t[:, None], dims="obs_id")

            # Seasonality components
            beta_fourier = pm.Normal(
                "beta_fourier", mu=self.model_config['beta_fourier']['mu'], sigma=self.model_config['beta_fourier']['sigma'], dims="fourier_features"
            )
            seasonality = pm.Deterministic(
                "seasonal", pm.math.dot(beta_fourier, fourier_terms), dims="obs_id"
            )

            # Combine components
            mu = trend + seasonality

            pm.Normal(
                "temperature_normalized",
                mu=mu,
                sigma=error,
                observed=self.y_normalized,
                dims="obs_id",
                )

    def sample_posterior_predictive(self, n_ahead: int, extend_idata: bool, combined: bool, **kwargs):
        self._data_setter(n_ahead)

        with pm.Model() as self.model:
            self.model.add_coord("obs_id", self.train_time_range)
            self.model.add_coord(
                "fourier_features",
                np.arange(len(self.train_fourier_series.to_numpy().T)),
            )

            t = pm.Data("time_range", self.train_time_range, dims="obs_id")
            fourier_terms = pm.Data(
                "fourier_terms", self.train_fourier_series.to_numpy().T
            )

            error = pm.HalfNormal("error", self.model_config['error'])

            # Trend component
            amplitude_trend = pm.HalfNormal("amplitude_trend", self.model_config['amplitude_trend'])
            ls_trend = pm.Gamma("ls_trend", alpha=self.model_config['ls_trend']['alpha'], beta=self.model_config['ls_trend']['beta'])
            cov_trend = amplitude_trend * pm.gp.cov.ExpQuad(1, ls_trend)

            gp_trend = pm.gp.HSGP(
                m=[10],
                c=5.,
                cov_func=cov_trend
            )
            trend = gp_trend.prior("trend", X=t[:, None], dims="obs_id")

            # Seasonality components
            beta_fourier = pm.Normal(
                "beta_fourier", mu=self.model_config['beta_fourier']['mu'], sigma=self.model_config['beta_fourier']['sigma'], dims="fourier_features"
            )
            seasonality = pm.Deterministic(
                "seasonal", pm.math.dot(beta_fourier, fourier_terms), dims="obs_id"
            )

            # Combine components
            mu = trend + seasonality

            pm.Normal(
                "temperature_normalized",
                mu=mu,
                sigma=error,
                observed=self.y_normalized,
                dims="obs_id",
                )

            self.model.add_coords({"obs_id_fut": np.arange(self.start, self.end, 1)})

            t_fut = pm.Data("time_range_fut", np.arange(self.start, self.end, 1))
            fourier_terms_fut = pm.Data("fourier_terms_fut", self.test_fourier_series.to_numpy().T)

            # Trend future component
            trend_fut = gp_trend.conditional("trend_fut", Xnew=t_fut[:, None], dims="obs_id_fut")

            # Seasonality components
            seasonality_fut = pm.Deterministic(
                "seasonal_fut", pm.math.dot(beta_fourier, fourier_terms_fut), dims="obs_id_fut"
            )

            mu_fut = trend_fut + seasonality_fut

            pm.Normal(
                "temperature_normalized_fut",
                mu=mu_fut,
                sigma=error,
                dims="obs_id_fut",
                )

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata, var_names=[f"{self.output_var}_normalized_fut"], predictions=True, **kwargs)
            if extend_idata:
                self.idata.extend(post_pred, join="right")

        posterior_predictive_samples = az.extract(
            post_pred, "predictions", combined=combined
        )

        return posterior_predictive_samples

    def fit(self, X: pl.DataFrame, y: pl.Series, normalize_target: bool = False) -> az.InferenceData:
        """
        Fits the model to the provided dataset
        ---
        Params:
            X: The dataset container predictor variables
            y: The target variable
            normalize_target: Whether to Z normalize the target variable
        """

        self.build_model(X, y, normalize_target=normalize_target)
        self.idata = self.sample_model(**self.sampler_config)

        X_df = X.to_pandas()
        combined_data = pd.concat([X_df, y.to_pandas()], axis=1)
        assert all(combined_data.columns), "All columns must have non-empty names"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=combined_data.to_xarray())  # type: ignore

        return self.idata  # type: ignore

    @staticmethod
    def get_default_model_config() -> dict:
        model_config = {
            "error": 0.2,
            "amplitude_trend": 1.0,
            "ls_trend": {"alpha": 48, "beta": 2},
            "beta_fourier": {"mu": 0, "sigma": 0.5},
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config= {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 4,
            "target_accept": 0.95,
        }
        return sampler_config

    @property
    def output_var(self):
        return "temperature"

    @property
    @abstractmethod
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        """
        Converts non-serializable values from model_config to their serializable reversable equivalent.
        Data types like pandas DataFrame, Series or datetime aren't JSON serializable,
        so in order to save the model they need to be formatted.

        Returns
        -------
        model_config: dict
        """
        return self.model_config

    def evaluate(self, y_true: pl.Series, forecasts: xr.Dataset, back_transform: bool = False) -> dict:
        """
        Evaluate our forecasts posterior predictive mean using the root mean squared error (RMSE) as the metric and evaluate our highest density interval's (HDI)s coverage
        ---
        Params:
            y_true: The ground truth temperatures
            forecasts: The forecasts
            back_transform: Whether we need to transform our forecasts back to the original scale
        """
        if back_transform:
            try:
                y_mean = self.y_mean
                y_std = self.y_std
            except AttributeError:
                y_mean = self.idata.attrs['y_mean']
                y_std = self.idata.attrs['y_std']
            posterior_predictive_mean = forecasts[f'{self.output_var}_normalized_fut'].mean(("chain", "draw")).values * y_std + y_mean
            hdi = az.hdi(forecasts[f'{self.output_var}_normalized_fut'], hdi_prob=0.8) * y_std + y_mean
        else:
            posterior_predictive_mean = forecasts[f'{self.output_var}_normalized_fut'].mean(("chain", "draw")).values
            hdi = az.hdi(forecasts[f'{self.output_var}_normalized_fut'], hdi_prob=0.8)

        error = y_true.to_numpy() - posterior_predictive_mean
        RMSE = np.sqrt(
            np.nanmean(
                np.square(error)
            )
        )

        coverage_df = pl.DataFrame(
            {
                "hdi_lower": hdi[f'{self.output_var}_normalized_fut'][:, 0].values,
                "hdi_upper": hdi[f'{self.output_var}_normalized_fut'][:, 1].values,
                "y_true": y_true
            }
        )

        COVERAGE = (
            coverage_df
              .filter(
                  pl.col("y_true").is_not_null()
              )
              .with_columns(
                  pl.when(
                      (pl.col("y_true") <= pl.col("hdi_upper")) &
                      (pl.col("y_true") >= pl.col("hdi_lower"))
                  )
                    .then(1.)
                    .otherwise(0.)
                    .alias("coverage")
              )
              .select(pl.col("coverage").mean()).item()
        )

        return {"RMSE": RMSE, "HDI_COVERAGE": COVERAGE}

    def _save_input_params(self, idata: az.InferenceData) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.
        """
        idata.attrs["city"] = self.city
        idata.attrs["y_mean"] = self.y_mean
        idata.attrs["y_std"] = self.y_std
