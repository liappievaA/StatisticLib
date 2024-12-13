#ifndef ARITHMETIC_BROWNIAN_MOTION_H
#define ARITHMETIC_BROWNIAN_MOTION_H

#include "DynamicsModel.h"
#include <Eigen/Dense>
#include <random>

namespace statisticslib {
namespace models {

class ArithmeticBrownianMotion : public DynamicsModel {
public:
    ArithmeticBrownianMotion(double volatility = 1.0, int timeStepCount = 1);
    ~ArithmeticBrownianMotion() override = default;

    Eigen::VectorXd simulateTrajectory(int backtestHorizon) override;
    Eigen::MatrixXd getReturnCorrelation(int h, int d) override;
    Eigen::VectorXd transformToStationary(const Eigen::VectorXd& returns, int backtestHorizon) override;
    Eigen::VectorXd transformToUniform(const Eigen::VectorXd& observations, int backtestHorizon) override;

private:
    double _volatility;
    double _businessDay;

    std::default_random_engine _rng;
    std::normal_distribution<double> _normalDist;
};

} // namespace models
} // namespace statisticslib

#endif // ARITHMETIC_BROWNIAN_MOTION_H