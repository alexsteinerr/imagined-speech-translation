#pragma once

#include "layers.h" 
#include <vector>
#include <string>
#include <random>

#include "iostream"

class Linear : public Layer {
public:
    Linear(int input_size, int output_size, std::string activation="", bool is_output=false)
        : Layer(activation) {
            this->input_size = input_size;
            this->output_size = output_size;
            this->is_output = is_output;

            double std_dev = sqrt(2.0 / input_size);

            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<double> distribution(0.0, std_dev);

            // Initialize weights and biases
            weights.resize(output_size, std::vector<double>(input_size));
            biases.resize(output_size);

            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    weights[i][j] = distribution(generator);
                }
                biases[i] = distribution(generator);
            }
        }

    std::vector<Neuron> forward(const std::vector<Neuron>& input_neurons) override;
    std::vector<double> backward(std::vector<double> &dout, double learning_rate);
    std::vector<std::vector<Neuron>> forward(const std::vector<std::vector<Neuron>> &input_neurons) override;

    std::vector<Neuron> input_neurons;
    std::vector<Neuron> output_neurons;


    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    
    int input_size;
    int output_size;

    bool is_output;
};
