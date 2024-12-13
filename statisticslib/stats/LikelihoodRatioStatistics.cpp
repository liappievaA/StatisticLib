#include "LikelihoodRatioStatistics.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <vector>

LikelihoodRatioStatistics::LikelihoodRatioStatistics(DynamicsModel& testedModel, int timeStep, int backtestHorizon)
    : Statistics(testedModel, timeStep, backtestHorizon) {
    // Get the return correlation matrix
    Eigen::MatrixXd correlationMatrix = testedModel.getReturnCorrelation(backtestHorizon, timeStep);
    // Invert the correlation matrix
    _inverseCorrelationMatrix = correlationMatrix.inverse();
}

double LikelihoodRatioStatistics::getValue(const std::vector<double>& observations, bool isModified) {
    // Transform observations to stationary
    std::vector<double> zSeries = _testedModel.transformToStationary(observations, _backtestHorizon);
    int n = zSeries.size();

    double estimatingMean, estimatingVar;

    if (isModified) {
        // Convert to Eigen vector
        Eigen::VectorXd zEigen(n);
        for(int i=0; i<n; ++i) {
            zEigen(i) = zSeries[i];
        }
        // Compute estimatingMean
        Eigen::VectorXd invCorrZ = _inverseCorrelationMatrix * zEigen;
        double sumInvCorr = _inverseCorrelationMatrix.sum();
        estimatingMean = invCorrZ.sum() / sumInvCorr;

        // Compute estimatingVar
        Eigen::VectorXd dev = zEigen - estimatingMean;
        estimatingVar = ( (_inverseCorrelationMatrix * dev).dot(dev) ) / n;
    } else {
        // Compute sample mean
        double mean = std::accumulate(zSeries.begin(), zSeries.end(), 0.0) / n;
        // Compute sample variance
        estimatingVar = (1.0 / n) * std::inner_product(zSeries.begin(), zSeries.end(), zSeries.begin(), 0.0,
            std::plus<double>(), [mean](double a, double b){ return (a - mean) * (b - mean); });
    }

    // Avoid log of zero or negative values
    if (estimatingVar <= 0.0) {
        estimatingVar = 1e-15;
    }

    // Compute likelihood ratio statistic
    double statistic = -n * (1.0 - estimatingVar + std::log(estimatingVar));
    return statistic;
}