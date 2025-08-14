#pragma once

#include "layers.h" 
#include <vector>
#include <string>
#include <random>
#include <iostream>

class Convolution : public Layer {
public:
    Convolution(int input_width, int input_height, int kernel_width, int kernel_height, std::string activation="");

    std::vector<std::vector<Neuron>> forward(const std::vector<std::vector<Neuron>>& input_neurons) override;
    std::vector<std::vector<Neuron>> backward(const std::vector<std::vector<Neuron>>& output_gradient, double learning_rate);

    // Function to correlate input and kernel
    std::vector<std::vector<double>> correlate2d(const std::vector<std::vector<Neuron>>& input, const std::vector<std::vector<double>>& kernel);

    int input_width;
    int input_height;

    int output_width;
    int output_height;

    int kernel_height;
    int kernel_width;

    std::vector<std::vector<double>> kernel; 
    std::vector<double> biases; 
    std::vector<std::vector<Neuron>> input_neurons;
    std::vector<std::vector<Neuron>> output_neurons; 

    std::vector<Neuron> forward(const std::vector<Neuron>& input_neurons) override;
};
