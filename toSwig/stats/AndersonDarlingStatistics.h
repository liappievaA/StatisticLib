#ifndef ANDERSON_DARLING_STATISTICS_H
#define ANDERSON_DARLING_STATISTICS_H

#include "Statistics.h"
#include <Eigen/Dense>
#include <boost/math/distributions/non_central_chi_squared.hpp>

namespace statisticslib {
namespace stats {

class AndersonDarlingStatistics : public Statistics {
public:
    AndersonDarlingStatistics(DynamicsModel& testedModel, int timeStep, int backtestHorizon);
    ~AndersonDarlingStatistics() override = default;

    double getValue(const Eigen::VectorXd& observations, bool isModified = true) override;

private:
    Eigen::VectorXd _eigenvalues;
    Eigen::MatrixXd _eigenvectors;
};

} // namespace stats
} // namespace statisticslib

#endif // ANDERSON_DARLING_STATISTICS_H