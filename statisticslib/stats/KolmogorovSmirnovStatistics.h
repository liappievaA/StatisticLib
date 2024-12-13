#ifndef KOLMOGOROV_SMIRNOV_STATISTICS_H
#define KOLMOGOROV_SMIRNOV_STATISTICS_H

#include "Statistics.h"
#include <vector>
#include <algorithm>
#include <cmath>

class KolmogorovSmirnovStatistics : public Statistics {
public:
    KolmogorovSmirnovStatistics(DynamicsModel& testedModel, int timeStep, int backtestHorizon);
    ~KolmogorovSmirnovStatistics() override = default;
    double getValue(const std::vector<double>& observations, bool isModified = true) override;

private:
    DynamicsModel& _testedModel;
    int _backtestHorizon;
};

#endif // KOLMOGOROV_SMIRNOV_STATISTICS_H