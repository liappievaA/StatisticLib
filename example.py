from statisticslib.models.ArithmeticBrownianMotion import ArithmeticBrownianMotion
from statisticslib.stats.KolmogorovSmirnovStatistics import KolmogorovSmirnovStatistics
from statisticslib.stats.AndersonDarlingStatistics import AndersonDarlingStatistics
from statisticslib.stats.LikelihoodRatioStatistics import LikelihoodRatioStatistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


N_obs = 1251
h = 1
d = 1

sigma_one = 1.0
sigma_three = 1.10

model_one = ArithmeticBrownianMotion(sigma_one, N_obs)
model_three = ArithmeticBrownianMotion(sigma_three, N_obs)

model_one.simulateTrajectory(h).plot()


def density_sample_series(data: pd.Series, step_for_range: int = 1):
    """Get hist plot of your sample.

    Args:
        step_for_range (int): amount of bins for hist
    """
    left = int(data.min())
    right = int(data.max()) + 1
    
    ranges = np.linspace(left, right, step_for_range)
    
    df_result = data.value_counts(bins=ranges, sort=False, normalize=True)

    df_result.index = ranges[:-1]

    return df_result

_, ax = plt.subplots(figsize=(15, 5), dpi=150)
density_sample_series(model_one.transformToUniform(model_one.simulateTrajectory(h), h), 90).plot(ax = ax)
density_sample_series(model_one.transformToUniform(model_three.simulateTrajectory(h), h), 90).plot(ax = ax)

KolmogorovSmirnovStatistics(model_one, d, h).plotDistribution({str(sigma_three) : model_three}, 120)
KolmogorovSmirnovStatistics(model_one, d, h).plotDiscriminatoryCurve(model_three)

AndersonDarlingStatistics(model_one, d, h).plotDistribution({str(sigma_three) : model_three}, 20)
AndersonDarlingStatistics(model_one, d, h).plotDiscriminatoryCurve(model_three)

LikelihoodRatioStatistics(model_one, d, h).plotDistribution({str(sigma_three) : model_three}, 30)
LikelihoodRatioStatistics(model_one, d, h).plotDiscriminatoryCurve(model_three)

