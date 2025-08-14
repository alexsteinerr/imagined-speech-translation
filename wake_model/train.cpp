#include "model/model.h"
#include "layers/losses.h"
#include "dataset/dataset.h"

#include "layers/linear.h"
#include "layers/convolution.h"
#include "layers/maxpool.h"
#include "layers/flatten.h"

#include "helper.h"

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>

int main() {
    Dataset dataset;
    dataset.load_dataset();

    std::cout << "Dataset Loaded!" << std::endl;

    int dataset_size = dataset.data.size();

    // CNN layers 
    Convolution conv(dataset.data[0].size(), 2, 32, 1, "relu");
    MaxPool max_pool(conv.output_width, conv.output_height,2,1);

    Convolution conv_2(max_pool.output_width, max_pool.output_height, 64, 1, "relu");
    MaxPool max_pool_2(conv_2.output_width, conv_2.output_height, 2,1);

    Convolution conv_3(max_pool_2.output_width, max_pool_2.output_height, 128, 1, "relu");
    MaxPool max_pool_3(conv_3.output_width, conv_3.output_height, 2,1);

    Flatten flatten;

    Linear positional_linear(conv_3.output_width * conv_3.output_height, 1024, "relu"); 
    Linear positional_output(positional_linear.output_size, dataset.data[0].size(), "softmax", true);

    // Model
    Model model;

    // Add layers to the model
    model.layers.push_back(std::make_unique<Convolution>(conv));
    model.layers.push_back(std::make_unique<MaxPool>(max_pool));

    model.layers.push_back(std::make_unique<Convolution>(conv_2));
    model.layers.push_back(std::make_unique<MaxPool>(max_pool_2));

    model.layers.push_back(std::make_unique<Convolution>(conv_3));
    model.layers.push_back(std::make_unique<MaxPool>(max_pool_3));

    model.layers.push_back(std::make_unique<Flatten>(flatten));
    model.layers.push_back(std::make_unique<Linear>(positional_linear));
    model.layers.push_back(std::make_unique<Linear>(positional_output));

    // Learning rate and epochs
    double learning_rate = 0.1;
    int epochs = 1000;
    int steps = 1;

    std::cout << "Starting Training!" << std::endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double cumulative_loss = 0.0;
        for(int step = 0; step < steps; step++){
            for (int i = 0; i < dataset_size; i++) {
                // Input and intermediate layers data
                std::vector<std::vector<Neuron>> hidden_layer_2d = dataset.data[i];
                std::vector<Neuron> hidden_layer_1d;
                std::vector<std::vector<Neuron>> outputs;

                // Forward pass
                for (const auto& layer : model.layers) {
                    if (auto* flatten_layer_ptr = dynamic_cast<Flatten*>(layer.get())) {
                        hidden_layer_1d = flatten_layer_ptr->forward_flatten(hidden_layer_2d);
                    } else if (auto* linear_layer_ptr = dynamic_cast<Linear*>(layer.get())) {
                        hidden_layer_1d = linear_layer_ptr->forward(hidden_layer_1d);
                        if (linear_layer_ptr->is_output) {
                            outputs.push_back(hidden_layer_1d);
                        }
                    } else if (auto* conv_layer_ptr = dynamic_cast<Convolution*>(layer.get())) {
                        hidden_layer_2d = conv_layer_ptr->forward(hidden_layer_2d);
                    } else if (auto* max_pool_layer_ptr = dynamic_cast<MaxPool*>(layer.get())) {
                        hidden_layer_2d = max_pool_layer_ptr->forward(hidden_layer_2d);
                    }
                }

                // Extract final output for loss calculation
                if (!outputs.empty()) {
                    std::vector<double> output_results;
                    for (const auto& neuron : outputs.back()) {
                        output_results.push_back(neuron.output);
                    }

                    // Calculate ground truth and loss
                    std::vector<double> ground_truth(output_results.size(), 0.0);
                    ground_truth[int(dataset.detection_time_rel[i] / 7)] = 1.0;
                    
                    cumulative_loss += categorical_cross_entropy_loss(output_results, ground_truth);
                    std::vector<double> derivative_loss = derivative_categorical_cross_entropy(output_results, ground_truth);

                    std::cout<<output_results[int(dataset.detection_time_rel[i] / 7)]<<std::endl;
                    std::cout << "Output: " << std::distance(output_results.begin(), std::max_element(output_results.begin(), output_results.end())) * 7
                            << " Expected: " << std::distance(ground_truth.begin(), std::max_element(ground_truth.begin(), ground_truth.end())) * 7 << std::endl;

                    // Backpropagation
                    std::vector<double> delta = derivative_loss;

                    // Start backpropagation from the output layer and move backward through each layer
                    for (int j=model.layers.size() - 1; j>=0; j--) {
                        auto& layer = model.layers[j];

                        if (auto* linear_layer_ptr = dynamic_cast<Linear*>(layer.get())) {
                            delta = linear_layer_ptr->backward(delta, learning_rate);
                        } else if (auto* flatten_layer_ptr = dynamic_cast<Flatten*>(layer.get())) {
                            // Convert the gradient from 1D to 2D for the convolutional layers
                            hidden_layer_2d = convert_to_2d_gradient(delta, conv_3.output_height, conv_3.output_width);
                        } else if (auto* conv_layer_ptr = dynamic_cast<Convolution*>(layer.get())) {
                            auto delta_2d = conv_layer_ptr->backward(hidden_layer_2d, learning_rate);
                            // Flatten the 2D gradient for further backpropagation if necessary
                            delta = flatten_gradient(delta_2d);
                        } else if (auto* maxpool_layer_ptr = dynamic_cast<MaxPool*>(layer.get())) {
                            auto delta_2d = maxpool_layer_ptr->backward(hidden_layer_2d);
                            hidden_layer_2d = delta_2d;
                        }
                    }
                }
            }
        }

        // Average loss per epoch
        std::cout << std::endl << "Epoch: " << epoch + 1 << " - Loss: " << cumulative_loss / (dataset_size*steps) << std::endl << std::endl;

        // Apply learning rate decay if needed
        dataset.shuffle_dataset();
    }

    model.save_model("seismic_test.bin");

    return 0;
}