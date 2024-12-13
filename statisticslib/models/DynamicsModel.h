#ifndef DYNAMICS_MODEL_H
#define DYNAMICS_MODEL_H

#include <vector>
#include <Eigen/Dense>

namespace statisticslib {
namespace models {

class DynamicsModel {
public:
    DynamicsModel(int timeStepCount);
    virtual ~DynamicsModel() = default;

    virtual Eigen::VectorXd simulateTrajectory(int backtestHorizon) = 0;
    virtual Eigen::MatrixXd getReturnCorrelation(int h, int d) = 0;
    virtual Eigen::VectorXd transformToStationary(const Eigen::VectorXd& returns, int backtestHorizon) = 0;
    virtual Eigen::VectorXd transformToUniform(const Eigen::VectorXd& observations, int backtestHorizon) = 0;

protected:
    int _timeStepCount;

    double _isValidValueNegative(double value);
};

} // namespace models
} // namespace statisticslib

#endif // DYNAMICS_MODEL_H