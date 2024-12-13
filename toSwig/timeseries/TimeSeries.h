#pragma once

#include <vector>
#include <memory>
#include "DynamicsModel.h"
#include "TimeSeries.h"

namespace statisticslib {
namespace stats {

class Statistics {
public:
    Statistics(std::shared_ptr<DynamicsModel> testedModel, int timeStep, int backtestHorizon);
    virtual ~Statistics() = default;

    virtual double getValue(const TimeSeries& observations, bool isModified = true) = 0;

    std::vector<double> getStatisticsValues(std::shared_ptr<DynamicsModel> model, int simulations = 100);

protected:
    std::shared_ptr<DynamicsModel> _testedModel;
    int _timeStep;
    int _backtestHorizon;
    int _simulationCount = 100;
};

} // namespace stats
} // namespace statisticslib