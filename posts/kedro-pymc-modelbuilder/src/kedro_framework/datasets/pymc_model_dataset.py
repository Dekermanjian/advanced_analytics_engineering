from pathlib import Path, PurePosixPath
from typing import Any

from kedro.io import AbstractDataset

from kedro_framework.pipelines.ML.ts_model import TSModel


class PyMCModelDataset(AbstractDataset):
    """
    ``PyMCDataset`` loads / save PyMC models from a given filepath as a TSModel object.
    """

    def __init__(self, filepath: str):
        """Creates a new instance of PyMCDataset to load / save PyMC models for a given filepath.

        Args:
            filepath: The location of the model netcdf file to load / save data.
        """
        self._filepath = PurePosixPath(filepath)
        self._path = Path(self._filepath.parent)

    def load(self) -> TSModel:
        """Loads data from the netcdf file.

        Returns:
            loaded TSModel
        """
        model = TSModel.load(self._filepath)
        return model

    def save(self, model: TSModel) -> None:
        """Saves PyMC model to the specified filepath."""
        self._path.mkdir(parents=True,  exist_ok=True)
        model.save(self._filepath)

    def _describe(self) -> dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath)
