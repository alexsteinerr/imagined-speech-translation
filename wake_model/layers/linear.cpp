#include "linear.h"
#include <iostream>

// Forward Pass
std::vector<Neuron> Linear::forward(const std::vector<Neuron>& input_neurons) {
    this->input_neurons = input_neurons;

    std::vector<Neuron> output_neurons(this->output_size);
    for (int i = 0; i < output_size; i++) {
        Neuron& out_neuron = output_neurons[i];
        out_neuron.output = 0.0;

        // Weighted sum of inputs
        for (int j = 0; j < input_size; j++) {
            out_neuron.output += input_neurons[j].output * weights[i][j];
        }

        // Add bias
        out_neuron.output += biases[i];

        // Apply activation function if needed
        if (!activation.empty() && activation != "softmax") {
            out_neuron.output = apply_activation(out_neuron.output, activation);
        }
    }

    if (this->activation == "softmax") {
        std::vector<double> output_results;
        for (const auto& neuron : output_neurons) {
            output_results.push_back(neuron.output);
        }

        output_results = softmax(output_results);

        for (int i = 0; i < output_neurons.size(); i++) {
            output_neurons[i].output = output_results[i];
        }
    }

    this->output_neurons = output_neurons;

    return output_neurons;
}

// Backward Pass
std::vector<double> Linear::backward(std::vector<double> &dout, double learning_rate) {
    std::vector<double> dinput(input_size, 0.0);

    // Derivative through activation function
    if (!(this->activation == "softmax" || this->activation.empty())) {
        for (int i = 0; i < output_size; i++) {
            double derivative = derivative_of_activation(output_neurons[i].output, this->activation);
            dout[i] *= derivative;
        }
    }

    // Calculate gradients for weights and biases
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            // Calculate input gradient to propagate backward
            dinput[j] += weights[i][j] * dout[i];

            // Update weight based on input and gradient
            weights[i][j] -= learning_rate * (input_neurons[j].output * dout[i]);
        }

        // Update bias
        biases[i] -= learning_rate * dout[i];
    }

    return dinput;
}


// Virtual Forward Function for 2D Input (Placeholder)
std::vector<std::vector<Neuron>> Linear::forward(const std::vector<std::vector<Neuron>> &input_neurons) {
    return input_neurons;
}
