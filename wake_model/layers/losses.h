#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm> 

inline double categorical_cross_entropy_loss(const std::vector<double>& predictions, const std::vector<double>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss -= labels[i] * log(predictions[i] + 1e-15); // Add epsilon to prevent log(0)
    }
    return loss;
}

inline std::vector<double> derivative_categorical_cross_entropy(const std::vector<double>& predictions, const std::vector<double>& labels) {
    std::vector<double> gradients(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        gradients[i] = predictions[i] - labels[i];
    }
    return gradients;
}
