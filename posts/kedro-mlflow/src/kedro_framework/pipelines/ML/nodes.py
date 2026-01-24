"""
This is a boilerplate pipeline 'ML'
generated using Kedro 0.19.11
"""
import logging
from datetime import timedelta
from typing import Optional

import arviz as az
import plotly.graph_objects as go
import polars as pl
import xarray as xr

from kedro_framework.pipelines.ML.ts_model import TSModel

logger = logging.getLogger(__name__)

def train_test_split(data: pl.DataFrame, testing_window: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits time-series dataset into test and train splits
    ---
    Params:
        data: The time-series dataset
        testing_window: The size of the testing set
    """

    testing_dataset = (
        data
        .sort("city", "date")
        .group_by("city")
        .agg(
            pl.all().tail(testing_window)
        )
        .explode("temperature", "date")
        .sort("city", "date")
    )

    training_dataset = (
        data
        .sort("city", "date")
        .group_by("city")
        .agg(
            pl.all().slice(0, pl.len() - testing_window)
        )
        .explode("temperature", "date")
        .sort("city", "date")
    )

    return training_dataset, testing_dataset

def train(training_dataset: pl.DataFrame, model_config: dict, sampler_config: dict) -> dict:
    """
    Train time-series models
    ---
    Params:
        training_dataset: Training data
        model_config: Model Configurations
        sampler_config: MCMC sampler configurations
    """
    city_models = {}
    for city in training_dataset['city'].sort().unique(maintain_order=True):
        city_dataset = training_dataset.filter(pl.col("city")==city).sort("date")
        model = TSModel(model_config=model_config, sampler_config=sampler_config, city=city)
        model.fit(X=city_dataset['date'], y=city_dataset['temperature'], normalize_target=True)
        city_models[f"{city}_model"] = model

    return city_models

def forecast(model: TSModel, n_ahead: int, ground_truth: Optional[pl.DataFrame] = None) -> xr.Dataset:
    """
    Generates forecasts from trained time-series and produces evaluations if ground truth is passed in.
    ---
    Params:
        model: The trained time-series model
        n_ahead: The forecast horizon
        ground_truth: The actual values to be compared with the forecasts
    """
    forecasts = model.sample_posterior_predictive(n_ahead=n_ahead, extend_idata=True, combined=False)
    if ground_truth is not None:
        evaluations = model.evaluate(y_true=ground_truth.filter(pl.col("city")==model.idata.attrs['city'])["temperature"], forecasts=forecasts, back_transform=True)
        logger.info(f"{model.idata.attrs['city']} Evaluations: {evaluations}")
    return {"forecasts": forecasts, "evaluations": evaluations}

def forecast_plot(training_dataset: pl.DataFrame, testing_dataset: pl.DataFrame, model: TSModel, forecast: xr.Dataset) -> go.Figure:
    """
    Generates plot showing in-sample posterior predictive performance as well as out-of-sample forecasts
    ---
    Params:
        training_dataset: The training split
        testing_dataset: The testing split
        model: the trained model
        forecasts: the forecast from the trainined model
    """
    # City specific data
    city = model.idata.attrs['city']
    city_training_dataset = training_dataset.filter(pl.col("city")==city)
    city_testing_dataset = testing_dataset.filter(pl.col("city")==city)

    # Model fit posterior predictive mean and HDI
    posterior_mean_normalized = model.idata.posterior_predictive['temperature_normalized'].mean(('chain', 'draw'))
    hdi_normalized = az.hdi(model.idata.posterior_predictive['temperature_normalized'], hdi_prob=0.8)
    posterior_mean = posterior_mean_normalized * model.idata.attrs['y_std'] + model.idata.attrs['y_mean']
    hdi = hdi_normalized * model.idata.attrs['y_std'] + model.idata.attrs['y_mean']

    # Forecast posterior predictive mean and HDI
    posterior_predictive_mean_normalized = forecast['temperature_normalized_fut'].mean(('chain', 'draw'))
    posterior_predictive_hdi_normalized = az.hdi(forecast['temperature_normalized_fut'], hdi_prob=0.8)
    posterior_predictive_mean = posterior_predictive_mean_normalized * model.idata.attrs['y_std'] + model.idata.attrs['y_mean']
    posterior_predictive_hdi = posterior_predictive_hdi_normalized * model.idata.attrs['y_std'] + model.idata.attrs['y_mean']

    fig = go.Figure()
    fig.add_traces(
        [
            go.Scatter(
                name="",
                x=city_training_dataset["date"],
                y=hdi["temperature_normalized"][:, 1],
                mode="lines",
                marker=dict(color="#eb8c34"),
                line=dict(width=0),
                legendgroup="HDI",
                showlegend=False
            ),
            go.Scatter(
                name="80% HDI",
                x=city_training_dataset["date"],
                y=hdi["temperature_normalized"][:, 0],
                mode="lines", marker=dict(color="#eb8c34"),
                line=dict(width=0),
                legendgroup="HDI",
                fill='tonexty',
                fillcolor='rgba(235, 140, 52, 0.5)'
            ),
            go.Scatter(
                x = city_training_dataset["date"],
                y = city_training_dataset["temperature"],
                mode="markers",
                marker_color="#48bbf0",
                name="actuals",
                legendgroup="actuals"
            ),
            go.Scatter(
                x = city_training_dataset["date"],
                y = posterior_mean,
                marker_color="blue",
                name="posterior_mean",
                legendgroup="posterior_mean"
            ),
            go.Scatter(
                name="",
                x=city_testing_dataset["date"],
                y=posterior_predictive_hdi["temperature_normalized_fut"][:, 1],
                mode="lines",
                marker=dict(color="#eb8c34"),
                line=dict(width=0),
                legendgroup="HDI",
                showlegend=False
            ),
            go.Scatter(
                name="",
                x=city_testing_dataset["date"],
                y=posterior_predictive_hdi["temperature_normalized_fut"][:, 0],
                mode="lines", marker=dict(color="#eb8c34"),
                line=dict(width=0),
                legendgroup="HDI",
                fill='tonexty',
                fillcolor='rgba(235, 140, 52, 0.5)',
                showlegend=False
            ),
            go.Scatter(
                x = city_testing_dataset["date"],
                y = city_testing_dataset["temperature"],
                mode="markers",
                marker_color="#48bbf0",
                name="",
                legendgroup="actuals",
                showlegend=False
            ),
            go.Scatter(
                x = city_testing_dataset["date"],
                y = posterior_predictive_mean,
                mode="lines",
                marker_color="yellow",
                name="",
                legendgroup="posterior_mean",
                showlegend=False
            ),
        ]
    )
    fig.update_layout(
        title = f"{city.title()} Temperature Forecast",
        xaxis=dict(
                title="Date",
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(count=10, label="10y", step="year", stepmode="backward"),
                            dict(step="all", label="All"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
                rangeselector_font_color="black",
                rangeselector_activecolor="hotpink",
                rangeselector_bgcolor="lightblue",
                autorangeoptions=dict(clipmax=city_testing_dataset['date'].max() + timedelta(days=30), clipmin=city_training_dataset['date'].min() - timedelta(days=30))
            ),
        yaxis=dict(
            title="Temperature"
        )
    )
    return fig
