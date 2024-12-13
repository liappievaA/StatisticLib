#ifndef LIKELIHOOD_RATIO_STATISTICS_H
#define LIKELIHOOD_RATIO_STATISTICS_H

#include "Statistics.h"
#include <Eigen/Dense>

class LikelihoodRatioStatistics : public Statistics {
public:
    LikelihoodRatioStatistics(DynamicsModel& testedModel, int timeStep, int backtestHorizon);
    ~LikelihoodRatioStatistics() override = default;
    double getValue(const std::vector<double>& observations, bool isModified = true) override;

private:
    Eigen::MatrixXd _inverseCorrelationMatrix;
};

#endif // LIKELIHOOD_RATIO_STATISTICS_H