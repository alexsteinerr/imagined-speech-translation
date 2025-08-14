#pragma once

#include "layers.h"
#include <vector>
#include <string>
#include <iostream>
#include <limits>

class MaxPool : public Layer {
public:
    MaxPool(int input_width, int input_height, int pool_width, int pool_height, int stride = 1)
        : Layer("") {
            this->input_width = input_width;
            this->input_height = input_width;
            this->pool_width = pool_width;
            this->pool_height = pool_height;
            this->stride = stride;

            this->output_height = (input_height - pool_height) / stride + 1;
            this->output_width = (input_width - pool_width) / stride + 1;
        }

    std::vector<std::vector<Neuron>> forward(const std::vector<std::vector<Neuron>>& input_neurons) override;
    std::vector<std::vector<Neuron>> backward(const std::vector<std::vector<Neuron>>& output_gradient);

    int input_width;
    int input_height;

    int pool_height;
    int pool_width;
    int stride;

    int output_width;
    int output_height;

    std::vector<std::vector<Neuron>> input_neurons;
    std::vector<std::vector<Neuron>> output_neurons;
    std::vector<std::vector<std::pair<int, int>>> max_indices;

    std::vector<Neuron> forward(const std::vector<Neuron>& input_neurons) override;
};
