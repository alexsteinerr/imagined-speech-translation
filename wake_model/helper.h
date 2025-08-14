#pragma once
#include <vector>
#include "neuron/neuron.h"

// Flatten a 2D Neuron vector into a 1D double vector
std::vector<double> flatten_gradient(const std::vector<std::vector<Neuron>>& gradient_2d) {
    std::vector<double> flattened_gradient;
    for (const auto& row : gradient_2d) {
        for (const auto& neuron : row) {
            flattened_gradient.push_back(neuron.output);
        }
    }
    return flattened_gradient;
}

// 1D double vector gradient back to 2D Neuron vector for convolutional layers
std::vector<std::vector<Neuron>> convert_to_2d_gradient(const std::vector<double>& gradient_1d, int height, int width) {
    std::vector<std::vector<Neuron>> gradient_2d(height, std::vector<Neuron>(width));
    int index = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            gradient_2d[y][x].output = gradient_1d[index++];
        }
    }
    return gradient_2d;
}