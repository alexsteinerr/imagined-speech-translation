#include "flatten.h"
#include <iostream>

// Forward Pass: Flatten the 2D input into a 1D vector
std::vector<Neuron> Flatten::forward_flatten(const std::vector<std::vector<Neuron>>& input_neurons) {
    this->input_height = input_neurons.size();
    this->input_width = input_neurons[0].size();

    std::vector<Neuron> flattened_output;
    flattened_output.reserve(input_height * input_width);

    for (const auto& row : input_neurons) {
        for (const auto& neuron : row) {
            flattened_output.push_back(neuron);
        }
    }

    return flattened_output;
}

// Backward Pass: Reshape the 1D gradient back into the original 2D structure
std::vector<std::vector<Neuron>> Flatten::backward(const std::vector<Neuron>& gradient) {
    std::vector<std::vector<Neuron>> output_gradient(input_height, std::vector<Neuron>(input_width));

    size_t index = 0;
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            output_gradient[i][j].output = gradient[index++].output;
        }
    }

    return output_gradient;
}

// Virtual Functions
std::vector<std::vector<Neuron>> Flatten::forward(const std::vector<std::vector<Neuron>>& input_neurons){
    return input_neurons;
}
std::vector<Neuron> Flatten::forward(const std::vector<Neuron>& input_neurons){
    return input_neurons;
}