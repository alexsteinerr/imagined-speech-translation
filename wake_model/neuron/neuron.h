#pragma once

#include <cstdlib> 
#include <ctime>  

class Neuron {
public:
    Neuron() {
        weight =  ((double) rand() / RAND_MAX - 0.5) * 0.1;
        bias = ((double) rand() / RAND_MAX - 0.5) * 0.1;
        output = 0.0; 
    }

    double weight; 
    double bias;   
    double input; 
    double output;  
};
