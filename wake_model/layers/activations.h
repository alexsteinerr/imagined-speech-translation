#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>

static const double e = 2.718281828459045235360287471352;
static const double pi = 3.141592653589793238462643383279;

inline double sigmoid(double inp) {
    return 1 / (1 + std::pow(e, -inp));
}

inline double derivative_sigmoid(double inp) {
    return sigmoid(inp) * (1 - sigmoid(inp));
}

inline double tanh(double inp){
    return (std::pow(e, inp) - std::pow(e, -inp)) / (std::pow(e, inp) + std::pow(e, -inp));
}

inline double derivative_tanh(double inp) {
    double tanh_inp = tanh(inp);
    return 1 - tanh_inp * tanh_inp;
}

inline std::vector<double> softmax(const std::vector<double>& inputs) {
    std::vector<double> exp_values(inputs.size());
    double max_input = *std::max_element(inputs.begin(), inputs.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        exp_values[i] = std::exp(inputs[i] - max_input);
        sum_exp += exp_values[i];
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        exp_values[i] /= sum_exp;
    }
    return exp_values;
}

inline std::vector<double> softmax_derivative(const std::vector<double>& output, const std::vector<double>& output_gradient) {
    size_t n = output.size();
    std::vector<double> gradient(n, 0.0);

    // Compute the gradient using the Jacobian of the softmax
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                sum += output[i] * (1.0 - output[j]) * output_gradient[j];
            } else {
                sum -= output[i] * output[j] * output_gradient[j];
            }
        }
        gradient[i] = sum;
    }

    return gradient;
}

inline double relu(double x) {
    return std::max(0.0, x);
}

inline double derivative_relu(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

inline double apply_activation(double inp, std::string activation) {
    if (activation == "sigmoid") {
        return sigmoid(inp);
    }else if (activation == "tanh"){
        return tanh(inp);
    }else if (activation == "relu"){
        return relu(inp);
    }else if(activation == ""){
        return inp;
    }
    throw std::invalid_argument("Unknown activation function: " + activation);
}

inline double derivative_of_activation(double inp, std::string activation) {
    if (activation == "sigmoid") {
        return derivative_sigmoid(inp);
    }else if (activation == "tanh"){
        return derivative_tanh(inp);
    }else if (activation == "relu"){
        return derivative_relu(inp);
    }else if(activation == "" || activation == "softmax"){
        return inp;
    }
    throw std::invalid_argument("Unknown derivative activation function: " + activation);
}
