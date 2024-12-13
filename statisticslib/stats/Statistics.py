from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from statisticslib.timeseries.TimeSeries import TimeSeries
from statisticslib.models.DynamicsModel import DynamicsModel


class Statistics(ABC):

    def __init__(
        self,
        testedModel: DynamicsModel,
        timeStep: int,
        backtestHorizon: int
    ):
        self._testedModel: DynamicsModel  = testedModel
        self._timeStep: int = timeStep
        self._backtestHorizon: int = backtestHorizon
        self.__simulationCount: int = 100

    @abstractmethod
    def getValue(
        self,
        observations: TimeSeries,
        isModified: bool = True
    ) -> float:
        pass

    def plotDiscriminatoryCurve(self, alternativeModel: DynamicsModel) -> None:
        """ Method makes plot of Discriminatory power.
        True Positive Rate (“TPR”) curve (or “ROC” curve), which gives the 
        probability that the test statistic from the misspecified model will be
        within the tail of distribution from the correctly specified model. 
        To quantify this, start with the cumulative distribution function 
        of the Null model that is to be tested for misspecification and assume 
        that we also have from the distribution function from the alternative 
        model.

        ### Args:
            `alternativeModel` (DynamicsModel) - model 

        """
        xCumsumBenchmark = self.__generateStatisticsValue(
            self._testedModel
        ).sort_values(ignore_index=True).to_numpy()

        xCumsumAlternative = self.__generateStatisticsValue(
            alternativeModel
        ).sort_values(ignore_index=True).to_numpy()

        trpCurve = {}

        for value in range(0, self.__simulationCount):
            # ????? как назвать
            temp = xCumsumBenchmark[value]

            # noinspection PyUnusedLocal
            result = None
            if temp >= xCumsumAlternative[-1]:
                result = 1
            elif temp <= xCumsumAlternative[0]:
                result = 0
            else:
                result = np.argsort(
                    np.abs(xCumsumAlternative - temp)
                )[0] / self.__simulationCount

            trpCurve[1 - value/self.__simulationCount] = 1 - result

        _, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.plot(np.linspace(0, 1, 2), label="No Disc.")
        ax.plot(trpCurve.keys(), trpCurve.values())
        ax.set_title("TPR curve.", fontsize=6)
        ax.set_ylabel("True Positive Rate", fontsize=6)
        ax.set_xlabel("False Positive Rate", fontsize=6)

        ax.legend()
        plt.show()

    def plotDistribution(
        self,
        models: dict[str, DynamicsModel],
        bins: int
    ) -> None:
        """
        Plot density of massive sample statistics model

        ### Args:
            `models` (dict[str, DynamicsModel]) : dictionary of alternative model with key \n 
            `bins` (int) : amount of backets for density \n
        """

        _, ax = plt.subplots(figsize=(15, 5), dpi=150)

        self.__generateStatisticsValue(
            self._testedModel
        ).getHistogram(bins).plot(ax=ax, label=f"base")

        for modelName, model in models.items():
            self.__generateStatisticsValue(
                model
            ).getHistogram(bins).plot(ax=ax, label=f"sigma={modelName}")

        ax.set_title("The value of the test statistic.")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.show()

    def __generateStatisticsValue(self, model: DynamicsModel) -> TimeSeries:
        simulations = []
        for _ in range(self.__simulationCount):
            simulations.append(
                self.getValue(model.simulateTrajectory(
                    self._backtestHorizon), isModified=True
                )
            )
        return TimeSeries(simulations)
