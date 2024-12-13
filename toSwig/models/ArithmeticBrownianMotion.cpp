#include "ArithmeticBrownianMotion.h"
#include <cmath>
#include <algorithm>

namespace statisticslib {
namespace models {

ArithmeticBrownianMotion::ArithmeticBrownianMotion(double volatility, int timeStepCount)
    : DynamicsModel(timeStepCount),
      _volatility(_isValidValueNegative(volatility)),
      _businessDay(1.0 / 250.0),
      _rng(std::random_device{}()),
      _normalDist(0.0, 1.0) {}

Eigen::VectorXd ArithmeticBrownianMotion::simulateTrajectory(int backtestHorizon) {
    int n = _timeStepCount + 1;
    Eigen::VectorXd randomVariable(n);
    double sigma_sqrt_dt = _volatility * std::sqrt(_businessDay * backtestHorizon);

    for (int i = 0; i < n; ++i) {
        randomVariable[i] = _normalDist(_rng) * sigma_sqrt_dt;
    }

    Eigen::VectorXd trajectory = randomVariable.cumsum();
    return trajectory;
}

Eigen::MatrixXd ArithmeticBrownianMotion::getReturnCorrelation(int h, int d) {
    int n = _timeStepCount;
    Eigen::MatrixXd correlation(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int abs_diff = std::abs(i - j) * d;
            if (abs_diff < h) {
                correlation(i, j) = 1.0 - static_cast<double>(abs_diff) / h;
            } else {
                correlation(i, j) = 0.0;
            }
        }
    }

    return correlation;
}

Eigen::VectorXd ArithmeticBrownianMotion::transformToStationary(const Eigen::VectorXd& returns, int backtestHorizon) {
    Eigen::VectorXd diff_returns = returns.tail(returns.size() - 1) - returns.head(returns.size() - 1);
    double factor = _volatility * std::sqrt(_businessDay * backtestHorizon);
    return diff_returns / factor;
}

Eigen::VectorXd ArithmeticBrownianMotion::transformToUniform(const Eigen::VectorXd& observations, int backtestHorizon) {
    Eigen::VectorXd stationary = transformToStationary(observations, backtestHorizon);
    Eigen::VectorXd uniform(stationary.size());

    for (int i = 0; i < stationary.size(); ++i) {
        uniform[i] = 0.5 * (1.0 + std::erf(stationary[i] / std::sqrt(2.0)));
    }

    return uniform;
}

} // namespace models
} // namespace statisticslib