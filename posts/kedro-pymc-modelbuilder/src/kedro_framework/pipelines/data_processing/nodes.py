"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.11
"""

import re

import plotly.graph_objects as go
import polars as pl


def process_datasets(partitioned_dataset: dict) -> pl.DataFrame:
    """
    Combine all cities into one dataframe and unpivot the data into long format
    ---
    params:
        partitioned_dataset: Our data partiotioned into key (filename) value(load method) pairs
    """
    # Missing values encoding
    mv_code = 999.9

    # Add a city column to each partiotion so that when we merge them all together we can identify each city
    datasets = [
        v().with_columns(city=pl.lit(re.findall(r"(?<=_).*(?=\.)", k)[0]))
        for k, v in partitioned_dataset.items()
    ]

    df_merged = pl.concat(datasets)

    df_processed = (
        df_merged.drop("D-J-F", "M-A-M", "J-J-A", "S-O-N", "metANN")
        .rename({"YEAR": "year"})
        .collect()  # Need to collect because can't unpivot a lazyframe
        .unpivot(
            on=[
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ],
            index=["city", "year"],
            variable_name="month",
            value_name="temperature",
        )
        .with_columns(
            pl.col("month")
            .str.to_titlecase()
            .str.strptime(dtype=pl.Date, format="%b")
            .dt.month()
        )
        .with_columns(
            date=pl.date(year=pl.col("year"), month=pl.col("month"), day=1),
        )
        .with_columns(
            pl.when(
                pl.col("temperature")
                == mv_code  # This is how missing data is coded in the dataset
            )
            .then(None)
            .otherwise(pl.col("temperature"))
            .name.keep(),
            pl.col("city").str.to_titlecase(),
        )
        .drop("year", "month")
    )
    return df_processed


def timeseries_plot(processed_dataframe: pl.DataFrame) -> go.Figure:
    """
    Plots each Brazilian city temperature time series
    """
    fig = go.Figure()
    for city in (
        processed_dataframe.select("city").unique(maintain_order=True).to_series()
    ):
        fig.add_trace(
            go.Scatter(
                x=processed_dataframe.filter(pl.col("city") == city).sort("date")[
                    "date"
                ],
                y=processed_dataframe.filter(pl.col("city") == city).sort("date")[
                    "temperature"
                ],
                name=city,
                hovertemplate="<b>Date</b>: %{x}<br><b>Temperature</b>: %{y}",
            )
        )
    fig.update_layout(
        title="Temperature Measurements of Brazilian Cities",
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
        ),
        yaxis=dict(title="Temperature in Celsius"),
    )
    return fig
