import numpy as np
from statisticslib.timeseries.TimeSeries import TimeSeries
from statisticslib.stats.Statistics import Statistics
from statisticslib.models.DynamicsModel import DynamicsModel


class LikelihoodRatioStatistics(Statistics):

    def __init__(
        self,
        testedModel: DynamicsModel,
        timeStep: int,
        backtestHorizon: int
    ):
        super().__init__(testedModel, timeStep, backtestHorizon)
        self.__invese_matrix: np.ndarray = np.linalg.inv(
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
        zSeries: TimeSeries = self._testedModel.transformToStationary(
            observations, self._backtestHorizon
        ).to_numpy()
        sizeSeries = zSeries.shape[0]

        # Модификация
        if isModified:
            estimatingMean: float = np.multiply(
                self.__invese_matrix, zSeries
            ).sum() / self.__invese_matrix.sum()

            temp_matrix: np.ndarray = (
                (self.__invese_matrix * (zSeries - estimatingMean)) *
                (zSeries - estimatingMean).reshape(-1, 1)
            )

            estimatingVar: float = (1 / sizeSeries) * temp_matrix.sum()
        # Обычный LR
        else:
            estimatingVar = (
                (1 / sizeSeries) *
                ((zSeries**2).sum()) - ((1 / sizeSeries)*(zSeries.sum()))**2
            )

        if estimatingVar == 0.0:
            estimatingVar = 1e-15

        return -sizeSeries * (1 - estimatingVar + np.log(estimatingVar))
