#ifndef ARITHMETIC_BROWNIAN_MOTION_H
#define ARITHMETIC_BROWNIAN_MOTION_H

#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <numeric>
#include <stdexcept>

class ArithmeticBrownianMotion {
private:
    double volatility;
    int timeStepCount;
    double businessDay;

public:
    // Constructor
    ArithmeticBrownianMotion(double volatility = 1.0, int timeStepCount = 1);

    // Simulate trajectory of arithmetic Brownian motion
    std::vector<double> simulateTrajectory(int backtestHorizon);

    // Transform returns to stationary form
    std::vector<double> transformToStationary(const std::vector<double>& returns, int backtestHorizon);

    // Transform observations to uniform distribution
    std::vector<double> transformToUniform(const std::vector<double>& observations, int backtestHorizon);

    // Get return correlation matrix
    std::vector<std::vector<double> > getReturnCorrelation(int h, int d);
};

#endif // ARITHMETIC_BROWNIAN_MOTION_H
