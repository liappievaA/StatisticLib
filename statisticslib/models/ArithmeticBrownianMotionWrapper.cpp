#include "ArithmeticBrownianMotionWrapper.h"

ArithmeticBrownianMotion::ArithmeticBrownianMotion(double volatility, int timeStepCount)
    : volatility(volatility), timeStepCount(timeStepCount), businessDay(1.0 / 250.0) {}

std::vector<double> ArithmeticBrownianMotion::simulateTrajectory(int backtestHorizon) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, std::sqrt(backtestHorizon * businessDay));

    std::vector<double> randomVariable(timeStepCount + 1);
    for (int i = 0; i <= timeStepCount; ++i) {
        randomVariable[i] = volatility * d(gen);
    }

    std::vector<double> trajectory(randomVariable.size());
    std::partial_sum(randomVariable.begin(), randomVariable.end(), trajectory.begin());

    return trajectory;
}

std::vector<double> ArithmeticBrownianMotion::transformToStationary(const std::vector<double>& returns, int backtestHorizon) {
    if (returns.size() < 2) {
        throw std::invalid_argument("Returns vector must have at least 2 elements");
    }

    std::vector<double> stationary(returns.size() - 1);
    for (size_t i = 1; i < returns.size(); ++i) {
        stationary[i - 1] = (returns[i] - returns[i - 1]) / 
                            (volatility * std::sqrt(businessDay * backtestHorizon));
    }

    return stationary;
}

std::vector<double> ArithmeticBrownianMotion::transformToUniform(const std::vector<double>& observations, int backtestHorizon) {
    std::vector<double> stationary = transformToStationary(observations, backtestHorizon);

    std::vector<double> uniform(stationary.size());
    for (size_t i = 0; i < stationary.size(); ++i) {
        uniform[i] = 0.5 * (1.0 + std::erf(stationary[i] / std::sqrt(2.0)));
    }

    return uniform;
}

std::vector<std::vector<double> > ArithmeticBrownianMotion::getReturnCorrelation(int h, int d) {
    if (timeStepCount <= 0) {
        throw std::invalid_argument("timeStepCount must be positive");
    }

    std::vector<std::vector<double> > correlationMatrix(timeStepCount, std::vector<double>(timeStepCount, 0.0));

    for (int i = 0; i < timeStepCount; ++i) {
        for (int j = 0; j < timeStepCount; ++j) {
            int diff = std::abs(i - j) * d;
            if (diff < h) {
                correlationMatrix[i][j] = 1.0 - static_cast<double>(diff) / h;
            }
        }
    }

    return correlationMatrix;
}
