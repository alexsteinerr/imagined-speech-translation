#include "dataset.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <random>
#include <numeric>
#include <filesystem>

// Assuming Neuron is defined somewhere in dataset.h
void Dataset::load_dataset() {
    std::ifstream file(catalog);
    std::string line, word;
    std::vector<std::string> headers;

    // Variables for tracking the longest sequence
    int max_length = 0;

    if (file.good()) {
        std::getline(file, line);  // Read the header
        std::stringstream ss(line);
        while (std::getline(ss, word, ',')) {
            headers.push_back(word);
        }
    }

    double min_time_rel = std::numeric_limits<double>::max();
    double max_time_rel = std::numeric_limits<double>::lowest();
    double min_velocity = std::numeric_limits<double>::max();
    double max_velocity = std::numeric_limits<double>::lowest();

    // First pass: find min, max values and the longest sequence length
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> row;

        while (std::getline(ss, word, ',')) {
            row.push_back(word);
        }

        std::string filename = row[0];
        if (std::filesystem::exists("data/training/" + filename + ".csv")) {
            std::ifstream training_file("data/training/" + filename + ".csv");

            if (training_file.good()) {
                std::getline(training_file, line); // Skip header

                int sequence_length = 0;
                while (std::getline(training_file, line)) {
                    sequence_length++;
                }

                max_length = std::max(max_length, sequence_length / 7);  // Update max length
            }
        }
    }

    // Reset the file pointer for the second pass
    file.clear();
    file.seekg(0, std::ios::beg);
    std::getline(file, line);  // Skip the header

    // Second pass: normalize and pad the sequences
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> row;

        while (std::getline(ss, word, ',')) {
            row.push_back(word);
        }

        std::string filename = row[0];
        std::string mq_type = row[4]; 
        double time_rel = std::stod(row[2]);

        if (std::filesystem::exists("data/training/" + filename + ".csv")) {
            detection_time_rel.push_back(time_rel);

            std::ifstream training_file("data/training/" + filename + ".csv");
            std::vector<std::vector<Neuron>> current_data;

            if (training_file.good()) {
                std::getline(training_file, line);  // Skip the header

                int count = 0;
                double time_average = 0.0;
                double velocity_average = 0.0;

                while (std::getline(training_file, line)) {
                    std::stringstream data_stream(line);
                    std::vector<std::string> data;
                    std::string data_point;

                    while (std::getline(data_stream, data_point, ',')) {
                        data.push_back(data_point);
                    }

                    time_average += std::stod(data[1]);
                    velocity_average += std::stod(data[2]);

                    if (count % 7 == 6) {
                        Neuron time_neuron, velocity_neuron;
                        time_neuron.input = time_average / 7;
                        velocity_neuron.input = velocity_average / 7;

                        std::vector<Neuron> current_neurons = {time_neuron, velocity_neuron};
                        current_data.push_back(current_neurons);

                        time_average = 0.0;
                        velocity_average = 0.0;
                    }

                    count++;
                }

                while (current_data.size() < max_length) {
                    Neuron zero_time_neuron, zero_velocity_neuron;
                    zero_time_neuron.input = 0.0;
                    zero_velocity_neuron.input = 0.0;
                    current_data.push_back({zero_time_neuron, zero_velocity_neuron});
                }
            }

            data.push_back(current_data);
        }
    }
}

void Dataset::shuffle_dataset() {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<std::vector<Neuron>>> shuffled_data;
    std::vector<double> shuffled_detection_time_rel;

    for (int i : indices) {
        shuffled_data.push_back(data[i]);
        shuffled_detection_time_rel.push_back(detection_time_rel[i]);
    }

    data = shuffled_data;
    detection_time_rel = shuffled_detection_time_rel;
}
