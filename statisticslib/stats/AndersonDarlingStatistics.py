import numpy as np
import scipy
from statisticslib.timeseries.TimeSeries import TimeSeries
from statisticslib.stats.Statistics import Statistics
from statisticslib.models.DynamicsModel import DynamicsModel


class AndersonDarlingStatistics(Statistics):

    def __init__(
        self,
        testedModel: DynamicsModel,
        timeStep: int,
        backtestHorizon: int
    ):
        super().__init__(testedModel, timeStep, backtestHorizon)
        self.__eigenvalues, self.__eigenvectors = np.linalg.eig(
            self._testedModel.getReturnCorrelation(
                self._backtestHorizon, self._timeStep
            )
        )

    def getValue(
        self,
        observations: TimeSeries,
        isModified: bool = True
    ) -> float:
        """
        ### Args:
            `observations` (TimeSeries) - trajectory of process \n
            `isModified` (bool) - flag for usage modified statistic \n
        """
        # Модификация
        if isModified:
            zSeries: np.ndarray = self._testedModel.transformToStationary(
                observations, self._backtestHorizon
            ).to_numpy()

            eigenvalues: np.ndarray = 1.0 / np.sqrt(self.__eigenvalues)

            zDecorrelaredSeries: np.ndarray = (
                zSeries @ self.__eigenvectors) * eigenvalues

            uSeries: np.ndarray = scipy.stats.norm.cdf(zDecorrelaredSeries)
        else:
            uSeries: np.ndarray = self._testedModel.transformToUniform(
                observations, self._backtestHorizon
            ).to_numpy()

        sizeSeries: int = uSeries.shape[0]

        # обычный AD
        uSeriesSorted: np.ndarray = np.sort(uSeries)
        sequenceIntNum: np.ndarray = np.arange(1, sizeSeries + 1)

        temp: np.ndarray = uSeriesSorted.copy()
        temp[temp == 0.0] = 1e-15
        log_u_values: np.ndarray = np.log(temp)

        uSeriesSorted[uSeriesSorted == 1.0] = 1 - 1e-15
        log_1_minus_u_values: np.ndarray = np.log(1 - uSeriesSorted)

        massive_one = 2*sequenceIntNum - 1
        massive_two = 2*(sizeSeries - sequenceIntNum) + 1

        result_one: np.ndarray = log_u_values * massive_one
        result_two: np.ndarray = log_1_minus_u_values * massive_two

        return -sizeSeries - (result_one + result_two).mean()
