#pragma once

#include "vector"
#include <fstream>
#include <iostream>
#include "../layers/layers.h"
#include <memory>

class Model{
    public:
        Model();

        void save_model(std::string file_name);
        void load_from_file(const std::string &filename);

        std::vector<std::unique_ptr<Layer>> layers;
};