#include "AndersonDarlingStatistics.h"
#include <cmath>
#include <limits>

namespace statisticslib {
namespace stats {

AndersonDarlingStatistics::AndersonDarlingStatistics(DynamicsModel& testedModel, int timeStep, int backtestHorizon)
    : Statistics(testedModel, timeStep, backtestHorizon) {
    Eigen::MatrixXd corrMatrix = testedModel.getReturnCorrelation(backtestHorizon, timeStep);
    Eigen::EigenSolver<Eigen::MatrixXd> es(corrMatrix);
    _eigenvalues = es.eigenvalues().real();
    _eigenvectors = es.eigenvectors().real();
}

double AndersonDarlingStatistics::getValue(const Eigen::VectorXd& observations, bool isModified) {
    if (isModified) {
        Eigen::VectorXd zSeries = _testedModel.transformToStationary(observations, _backtestHorizon);
        Eigen::VectorXd eigenvaluesInv = 1.0 / std::sqrt(_eigenvalues);
        Eigen::VectorXd zDecorrelaredSeries = (_eigenvectors.transpose() * zSeries).array() * eigenvaluesInv.array();
        Eigen::VectorXd uSeries = 0.5 * (1.0 + erf(zDecorrelaredSeries.array() / std::sqrt(2.0)));
    } else {
        Eigen::VectorXd uSeries = _testedModel.transformToUniform(observations, _backtestHorizon);
    }

    int n = uSeries.size();
    Eigen::VectorXd uSeriesSorted = uSeries.sort();
    Eigen::VectorXd sequenceIntNum = Eigen::VectorXd::LinSpaced(n, 1, n);

    Eigen::VectorXd log_u_values = uSeriesSorted.array().log();
    log_u_values = log_u_values.replace(-std::numeric_limits<double>::infinity(), -1e15); // Handle log(0)
    Eigen::VectorXd log_1_minus_u_values = (1.0 - uSeriesSorted).array().log();
    log_1_minus_u_values = log_1_minus_u_values.replace(-std::numeric_limits<double>::infinity(), -1e15); // Handle log(0)

    Eigen::VectorXd massive_one = 2.0 * sequenceIntNum - 1.0;
    Eigen::VectorXd massive_two = 2.0 * (n - sequenceIntNum) + 1.0;

    Eigen::VectorXd result_one = log_u_values.array() * massive_one.array();
    Eigen::VectorXd result_two = log_1_minus_u_values.array() * massive_two.array();

    double mean_result = (result_one + result_two).mean();
    double statistic = -n - mean_result;
    return statistic;
}

} // namespace stats
} // namespace statisticslib