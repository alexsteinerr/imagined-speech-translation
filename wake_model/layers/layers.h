#pragma once
#include <vector>
#include <string>
#include "../neuron/neuron.h"
#include "activations.h"

class Layer {
public:
    Layer(std::string activation="", bool output=false)
        : activation(activation), output(output) {}

    virtual ~Layer() = default;  

    virtual std::vector<Neuron> forward(const std::vector<Neuron>& input_neurons) = 0;  
    virtual std::vector<std::vector<Neuron>> forward(const std::vector<std::vector<Neuron>>& input_neurons) = 0;  

    std::vector<Neuron> input;
    std::string activation;

    bool output;
};
