from __future__ import annotations

import numpy as np
import pandas as pd


class TimeSeries(pd.Series):

    def getHistogram(self, bins: int) -> pd.Series:
        leftEnd = int(self.min()) - 1
        rightEnd = int(self.max()) + 1

        binsRanges: np.ndarray = np.linspace(leftEnd, rightEnd, bins)

        countValuesAtSample: pd.Series = self.value_counts(
            bins=binsRanges,
            sort=False,
            normalize=True,
            dropna=True
        )
        trimZerosFromSample: np.ndarray = np.trim_zeros(
            countValuesAtSample.values
        )

        temp = countValuesAtSample.values != 0.0
        startIndex = temp.argmax()
        endIndex = temp.shape[0] - temp[::-1].argmax() - 1

        flagStart = 0
        flagEnd = 0

        if startIndex > 0:
            trimZerosFromSample = np.insert(trimZerosFromSample, 0, 0.0)
            flagStart = 1
        if endIndex < bins - 1:
            trimZerosFromSample = np.append(trimZerosFromSample, 0.0)
            flagEnd = 1

        return pd.Series(
            trimZerosFromSample,
            index=binsRanges[startIndex-flagStart:endIndex+flagEnd+1]
        )
