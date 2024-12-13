#include "KolmogorovSmirnovStatistics.h"
#include <algorithm>
#include <cmath>

KolmogorovSmirnovStatistics::KolmogorovSmirnovStatistics(DynamicsModel& testedModel, int timeStep, int backtestHorizon)
    : Statistics(testedModel, timeStep, backtestHorizon), _testedModel(testedModel), _backtestHorizon(backtestHorizon) {}

double KolmogorovSmirnovStatistics::getValue(const std::vector<double>& observations, bool isModified) {
    std::vector<double> uSeries;

    if (isModified) {
        uSeries = _testedModel.transformToUniform(observations, _backtestHorizon);
    } else {
        uSeries = observations;
    }

    std::sort(uSeries.begin(), uSeries.end());

    int n = uSeries.size();
    double D = 0.0;

    for (int i = 1; i <= n; ++i) {
        double empirical = static_cast<double>(i) / n;
        double theoretical = uSeries[i - 1];
        double diff = std::abs(empirical - theoretical);
        if (diff > D) {
            D = diff;
        }
    }

    double theoretical_start = 0.0;
    double empirical_start = 0.0;
    double diff_start = std::abs(empirical_start - theoretical_start);
    if (diff_start > D) {
        D = diff_start;
    }

    double theoretical_end = 1.0;
    double empirical_end = 1.0;
    double diff_end = std::abs(empirical_end - theoretical_end);
    if (diff_end > D) {
        D = diff_end;
    }

    return D;
}