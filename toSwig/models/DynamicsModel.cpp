#include "DynamicsModel.h"

namespace statisticslib {
namespace models {

DynamicsModel::DynamicsModel(int timeStepCount)
    : _timeStepCount(timeStepCount) {
    if (_timeStepCount <= 0) {
        throw std::invalid_argument("timeStepCount must be positive");
    }
}

double DynamicsModel::_isValidValueNegative(double value) {
    if (value <= 0.0) {
        throw std::invalid_argument("value is lower than zero");
    }
    return value;
}

} // namespace models
} // namespace statisticslib