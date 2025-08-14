#include <fstream>
#include <unordered_map>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <algorithm>  
#include <random>

#include "../neuron/neuron.h"

namespace fs = std::filesystem;

class Dataset {
public:
    std::string catalog = "data/catalog.csv";

    std::vector<std::string> filenames;
    
    std::vector<std::vector<std::vector<Neuron>>> data;
    std::vector<double> detection_time_rel;

    void load_dataset();
    void shuffle_dataset();
};