#include "Statistics.h"
#include "TimeSeries.h"

namespace statisticslib {
namespace stats {

Statistics::Statistics(std::shared_ptr<DynamicsModel> testedModel, int timeStep, int backtestHorizon)
    : _testedModel(testedModel), _timeStep(timeStep), _backtestHorizon(backtestHorizon) {}

std::vector<double> Statistics::getStatisticsValues(std::shared_ptr<DynamicsModel> model, int simulations) {
    std::vector<double> statisticsValues;
    for (int i = 0; i < simulations; ++i) {
        TimeSeries trajectory = model->simulateTrajectory(_backtestHorizon);
        double statValue = getValue(trajectory, true);
        statisticsValues.push_back(statValue);
    }
    return statisticsValues;
}

} // namespace stats
} // namespace statisticslib