import scipy
import numpy as np
from statisticslib.models.DynamicsModel import DynamicsModel
from statisticslib.timeseries.TimeSeries import TimeSeries


class ArithmeticBrownianMotion(DynamicsModel):
    """
    Simulate Arithmetic Brownian Motion which can define by SDE 
    dS = lambda sigma dW.

    ### Args:
        `volatility` (float) : value of volatility(const) sigma. \n
        `timeStepCount` (int) : amount of observed experiments. \n
    """

    def __init__(
        self,
        volatility: float = 1.0,
        timeStepCount: int = 1,
    ):
        super().__init__(timeStepCount)
        self.__volatility: float = self._isValidValueNegative(volatility)
        self.__businessDay: float = 1. / 250.

    def simulateTrajectory(self, backtestHorizon: int) -> TimeSeries:
        randomVariable = self.__volatility*np.random.normal(
            0, np.sqrt(backtestHorizon*self.__businessDay), self._timeStepCount + 1
        )

        return TimeSeries(np.cumsum(randomVariable))

    def transformToStationary(
            self, 
            returns: TimeSeries,
            backtestHorizon: int
        ) -> TimeSeries:
        return returns.diff().dropna() / (
            self.__volatility * np.sqrt(self.__businessDay*backtestHorizon)
        )

    def transformToUniform(
        self,
        observations: TimeSeries,
        backtestHorizon: int
    ) -> TimeSeries:
        return self.transformToStationary(
            observations, backtestHorizon
        ).map(scipy.stats.norm.cdf)

    def getReturnCorrelation(self, h: int, d: int) -> np.ndarray:
        vectorizeMakeCovarMatrix = np.vectorize(
            lambda i, j:
                (1 - (np.abs(i - j)*d) / h) if np.abs(i - j)*d < h else 0
        )

        return np.fromfunction(vectorizeMakeCovarMatrix,
                               (self._timeStepCount, self._timeStepCount)
                               )
