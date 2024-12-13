import scipy
import numpy as np
from statisticslib.models.DynamicsModel import DynamicsModel
from statisticslib.timeseries.TimeSeries import TimeSeries


class CoxIngersollRoss(DynamicsModel):
    """ 
    Simulate Arithmetic Cox-Ingesoll-Ross process which can define by SDE 
    $$dr_t = k(\theta - r_t)\dt + \sigma\sqrt{r_t}\dW_t&$$     
    This class realise Monte-Carlo methods of weighted Euler-Maruyama 
    Scheme.

    ### Args:
        `volatility` (float) : value of volatility(const) r\sigma. \n
        `meanRevert` (float) : value of volatility(const) r\sigma. \n
        `drift` (float) : value of volatility(const) r\sigma. \n
        `timeStepCount` (int) : amount of observed experiments. \n
        `startPoint` (float) : the point of begin path trajectory. \n
    """

    def __init__(
        self,
        volatility: float = 0.0001,
        meanRevert: float = 1.0,
        drift: float = 0.001,
        timeStepCount: int = 1,
        startPoint: float = 0.001
    ):
        super().__init__(timeStepCount)
        self.__volatility: float = self._isValidValueNegative(volatility)
        self.__meanRevert: float = self._isValidValueNegative(meanRevert)
        self.__asymptoticMean: float = self._isValidValueNegative(drift)
        self.__checkConditionForParam()

        self.__businessDay: float = 1. / 250.
        self.__startPoint: float = startPoint

    def simulateTrajectory(self, backtestHorizon: int) -> TimeSeries:
        path = np.zeros(self._timeStepCount + 1)
        path[0] = self.__startPoint

        randomVariable = scipy.stats.norm.rvs(
            loc=0,
            scale=np.sqrt(self.__businessDay*backtestHorizon),
            size=self._timeStepCount
        )
        for t in range(0, self._timeStepCount):

            path[t + 1] = path[t] + self.__meanRevert * \
                (self.__asymptoticMean - path[t]) * backtestHorizon * \
                self.__businessDay + self.__volatility * \
                np.sqrt(path[t]) * randomVariable[t]

            if path[t] is None:
                raise "Value is incoreect for the process"

        return TimeSeries(path)

    def getReturnCorrelation(self, h: int, d: int) -> np.ndarray:
        vectorizeMakeCovarMatrix = np.vectorize(
            lambda i, j:
                (1 - (np.abs(i - j)*d) / h) if np.abs(i - j)*d < h else 0
        )

        return np.fromfunction(vectorizeMakeCovarMatrix,
                               (self._timeStepCount, self._timeStepCount)
                               )

    def transformToStationary(
        self,
        returns: TimeSeries,
        backtestHorizon: int
    ) -> TimeSeries:
        return returns

    def transformToUniform(
        self,
        observations: TimeSeries,
        backtestHorizon: int
    ) -> TimeSeries:
        massive = np.zeros(observations.shape[0])

        for index, element in observations.items():
            if index == 0:
                continue
            T = index*self.__businessDay*backtestHorizon
            c = 2*self.__meanRevert / (self.__volatility**2*(
                1 - np.exp(-self.__meanRevert*T))
            )

            massive[index] = scipy.stats.ncx2.cdf(
                2*c*element,
                4*self.__meanRevert*self.__asymptoticMean / self.__volatility**2,
                2*self.__startPoint*c*np.exp(-self.__meanRevert*T)
            )
        return TimeSeries(massive)

    def _isValidValueNegative(
        self,
        value: float
    ) -> float:
        if value <= 0:
            raise ValueError("value is lower than zero")
        return value

    def __checkConditionForParam(self):
        if self.__meanRevert*self.__asymptoticMean < self.__volatility**2:
            raise ValueError(
                "Mistake for main condition for process parametrs"
            )
