#include "convolution.h"
#include <cmath>

// Constructor: Initializes the kernel weights and biases
Convolution::Convolution(int input_width, int input_height, int kernel_width, int kernel_height, std::string activation)
    : Layer(activation) {
    
    this->kernel_width = kernel_width;
    this->kernel_height = kernel_height;
    this->input_width = input_width;
    this->input_height = input_height;

    // Initialize kernel weights with He initialization
    this->kernel.resize(kernel_height, std::vector<double>(kernel_width, 0.0));
    double limit = sqrt(6.0 / (kernel_width * kernel_height));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-limit, limit);

    for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
            kernel[kh][kw] = dist(gen);
        }
    }

    this->output_height = input_height - this->kernel_height + 1;
    this->output_width = input_width - this->kernel_width + 1;

    // Initialize biases (one bias per filter)
    this->biases.resize(1, ((double) rand() / RAND_MAX - 0.5) * 0.1);
    this->output_neurons.resize(this->output_height, std::vector<Neuron>(this->output_width));
}

// Forward Pass: Convolve the input with the kernel
std::vector<std::vector<Neuron>> Convolution::forward(const std::vector<std::vector<Neuron>>& input_neurons) {
    this->input_neurons = input_neurons;

    std::vector<std::vector<Neuron>> output_neurons(output_height, std::vector<Neuron>(output_width));

    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            double sum = 0.0;

            for (int ky = 0; ky < kernel_height; ++ky) {
                for (int kx = 0; kx < kernel_width; ++kx) {
                    sum += input_neurons[y + ky][x + kx].output * kernel[ky][kx];
                }
            }

            // Add bias to the sum
            output_neurons[y][x].output = sum + biases[0];
        }
    }

    return output_neurons;
}

// Backward Pass: Calculate gradients and update kernel and bias
std::vector<std::vector<Neuron>> Convolution::backward(const std::vector<std::vector<Neuron>>& output_gradient, double learning_rate) {
    std::vector<std::vector<double>> kernel_gradient(kernel_height, std::vector<double>(kernel_width, 0.0));
    std::vector<std::vector<Neuron>> input_gradient(input_height, std::vector<Neuron>(input_width));

    // Initialize input gradients to zero
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            input_gradient[y][x].output = 0.0;
        }
    }

    // Calculate kernel and input gradients
    for (int y = 0; y < output_gradient.size(); ++y) {
        for (int x = 0; x < output_gradient[y].size(); ++x) {
            double dout = output_gradient[y][x].output;

            // Compute kernel gradients
            for (int ky = 0; ky < kernel_height; ++ky) {
                for (int kx = 0; kx < kernel_width; ++kx) {
                    kernel_gradient[ky][kx] += input_neurons[y + ky][x + kx].output * dout;
                }
            }

            // Compute input gradients - kernel flipped for correct gradient propagation
            for (int ky = 0; ky < kernel_height; ++ky) {
                for (int kx = 0; kx < kernel_width; ++kx) {
                    int input_y = y + ky;
                    int input_x = x + kx;

                    if (input_y < input_height && input_x < input_width) {
                        input_gradient[input_y][input_x].output += kernel[kernel_height - ky - 1][kernel_width - kx - 1] * dout;
                    }
                }
            }
        }
    }

    // Update kernel weights
    for (int ky = 0; ky < kernel_height; ++ky) {
        for (int kx = 0; kx < kernel_width; ++kx) {
            kernel[ky][kx] -= learning_rate * kernel_gradient[ky][kx];
        }
    }

    // Update bias
    double bias_gradient = 0.0;
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            bias_gradient += output_gradient[y][x].output;
        }
    }
    biases[0] -= learning_rate * bias_gradient;

    return input_gradient;
}

// Helper Function: Correlate Input and Kernel
std::vector<std::vector<double>> Convolution::correlate2d(const std::vector<std::vector<Neuron>>& input, const std::vector<std::vector<double>>& kernel) {
    int input_height = input.size();
    int input_width = input[0].size();
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    std::vector<std::vector<double>> output(output_height, std::vector<double>(output_width, 0.0));

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            double sum = 0.0;
            for (int m = 0; m < kernel_height; ++m) {
                for (int n = 0; n < kernel_width; ++n) {
                    sum += input[i + m][j + n].output * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

// Virtual functions
std::vector<Neuron> Convolution::forward(const std::vector<Neuron>& input_neurons){
    return input_neurons;
}