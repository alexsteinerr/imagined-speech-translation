#include "maxpool.h"
#include <cmath>
#include <limits>

// Forward Pass
std::vector<std::vector<Neuron>> MaxPool::forward(const std::vector<std::vector<Neuron>>& input_neurons) {
    this->input_neurons = input_neurons;

    // Resize output and max_indices to fit the expected output dimensions
    this->output_neurons.resize(output_height, std::vector<Neuron>(output_width));
    this->max_indices.resize(output_height, std::vector<std::pair<int, int>>(output_width));

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            double max_value = -std::numeric_limits<double>::infinity();
            int max_i = -1;
            int max_j = -1;

            // Iterate over the pooling window
            for (int m = 0; m < pool_height; ++m) {
                for (int n = 0; n < pool_width; ++n) {
                    int input_i = i * stride + m;
                    int input_j = j * stride + n;

                    // Ensure we're within bounds
                    if (input_i < input_height && input_j < input_width) {
                        double value = input_neurons[input_i][input_j].output;
                        if (value > max_value) {
                            max_value = value;
                            max_i = input_i;
                            max_j = input_j;
                        }
                    }
                }
            }

            output_neurons[i][j].output = max_value;
            max_indices[i][j] = std::make_pair(max_i, max_j);
        }
    }

    return output_neurons;
}

// Backward Pass
std::vector<std::vector<Neuron>> MaxPool::backward(const std::vector<std::vector<Neuron>>& output_gradient) {
    int input_height = input_neurons.size();
    int input_width = input_neurons[0].size();

    int output_height = output_gradient.size();
    int output_width = output_gradient[0].size();

    // Initialize input gradient with zeros
    std::vector<std::vector<Neuron>> input_gradient(input_height, std::vector<Neuron>(input_width));

    // Distribute the gradient to the positions of the maxima
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            int max_i = max_indices[i][j].first;
            int max_j = max_indices[i][j].second;

            if (max_i >= 0 && max_j >= 0) {
                input_gradient[max_i][max_j].output += output_gradient[i][j].output;
            }
        }
    }

    return input_gradient;
}

// Virtual Functions
std::vector<Neuron> MaxPool::forward(const std::vector<Neuron>& input_neurons){
    return input_neurons;
}