import numpy as np
import scipy
from statisticslib.timeseries.TimeSeries import TimeSeries
from statisticslib.stats.Statistics import Statistics
from statisticslib.models.DynamicsModel import DynamicsModel


class KolmogorovSmirnovStatistics(Statistics):

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

        uSeriesSorted: np.ndarray = np.sort(uSeries)
        sequenceIntNum: np.ndarray = np.arange(1, sizeSeries + 1)

        # ???? без понятия как переименовать
        d_plus: float = np.max((sequenceIntNum / sizeSeries) - uSeriesSorted)
        d_minus: float = np.max(
            uSeriesSorted - ((sequenceIntNum - 1) / sizeSeries)
        )

        return np.max([d_plus, d_minus])
