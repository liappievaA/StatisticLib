from abc import ABC, abstractmethod
import numpy as np
from statisticslib.timeseries.TimeSeries import TimeSeries


class DynamicsModel(ABC):
    def __init__(
        self,
        timeStepCount: int
    ):
        self._timeStepCount: int = self._isValidValueNegative(
            timeStepCount
        )

    @abstractmethod
    def simulateTrajectory(self, backtestHorizon: int) -> TimeSeries:
        pass

    @abstractmethod
    def getReturnCorrelation(self) -> np.ndarray:
        pass

    @abstractmethod
    def transformToUniform(
        self,
        observations: TimeSeries
    ) -> TimeSeries:
        pass

    def _isValidValueNegative(
        self,
        value: float
    ) -> float:
        if value <= 0:
            raise ValueError("value is lower than zero")
        return value
