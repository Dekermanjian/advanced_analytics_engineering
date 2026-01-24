from pathlib import Path, PurePosixPath
from typing import Any

import xarray as xr
from kedro.io import AbstractDataset


class PyMCForecastDataset(AbstractDataset):
    """
    ``PyMCForecastDataset`` loads / save PyMC forecasts from a given filepath as an xarray Dataset.
    """

    def __init__(self, filepath: str):
        """Creates a new instance of PyMCForecastDataset to load / save PyMC forecasts for a given filepath.

        Args:
            filepath: The location of the forecast netcdf file to load / save.
        """
        self._filepath = PurePosixPath(filepath)
        self._path = Path(self._filepath.parent)

    def load(self) -> xr.Dataset:
        """Loads data from the netcdf file.

        Returns:
            loaded forecasts
        """
        return xr.load_dataset(self._filepath)

    def save(self, forecast: xr.Dataset) -> None:
        """Saves PyMC forecasts to the specified filepath."""
        self._path.mkdir(parents=True,  exist_ok=True)
        forecast.to_netcdf(path=self._filepath)

    def _describe(self) -> dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath)
