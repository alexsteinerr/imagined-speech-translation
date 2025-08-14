#include "model.h"

Model::Model(){}

void Model::save_model(std::string file_name){
    std::ofstream out_file(file_name, std::ios::binary);
    if (out_file.is_open()) {
        out_file.write(reinterpret_cast<const char*>(this), sizeof(*this));
        out_file.close();
    } else {
        std::cerr << "Could not open file for writing\n";
    }
}

void Model::load_from_file(const std::string &filename) {
    std::ifstream in_file(filename, std::ios::binary);
    if (in_file.is_open()) {
        in_file.read(reinterpret_cast<char*>(this), sizeof(*this));
        in_file.close();
    } else {
        std::cerr << "Could not open file for reading\n";
    }
}