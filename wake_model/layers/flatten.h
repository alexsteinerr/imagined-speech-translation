#pragma once

#include "layers.h" 
#include <vector>
#include <string>
#include <iostream>

class Flatten : public Layer {
public:
    Flatten(std::string activation = "")
        : Layer(activation), input_height(0), input_width(0) {}

    std::vector<Neuron> forward_flatten(const std::vector<std::vector<Neuron>>& input_neurons);
    std::vector<std::vector<Neuron>> backward(const std::vector<Neuron>& gradient);

    int input_height;
    int input_width;

    std::vector<std::vector<Neuron>> forward(const std::vector<std::vector<Neuron>>& input_neurons) override;
    std::vector<Neuron> forward(const std::vector<Neuron>& input_neurons) override;
};
