#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <chrono>

#include "neuralnet.hpp"

#define INSTANCES_SIZE 1000
#define INPUTS_SIZE 10

using std::vector, std::cout, std::endl, std::unique_ptr;

int main( int argc, char** argv ){

	//Random double generator with clock as seed
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::uniform_real_distribution<double> unif(/*Lower bound:*/0.0,   /*Upper bound:*/1.0);
  	std::default_random_engine re (seed);

	//Vector of vector pointers to avoid a vector of vectors
	vector< vector<double>* > Inputs;
	Inputs.reserve(INPUTS_SIZE + 1); //Add one slot for the bias ones
	vector<double>* x = new vector<double>(INSTANCES_SIZE, 1.0); //Bias ones
	Inputs.emplace_back(x);//Set first input to ones for use with the bias

	//Fill the rest of the inputs with vectors of random numbers between 1.0 and 0.0
	for (int i = 0; i < INPUTS_SIZE; ++i){
		x = new vector<double>;
		x->reserve(INSTANCES_SIZE);
		Inputs.emplace_back(x);
	}

	//Fill with random doubles, skip first input because its ones for the bias
	for (int i = 1; i < INPUTS_SIZE + 1; ++i){
		for (int j = 0; j < INSTANCES_SIZE; ++j){
			Inputs[i]->emplace_back(unif(re));
		}
	}

	//Generate labels
	vector<double> Labels;
	Labels.reserve(INSTANCES_SIZE);
	for (int i = 0; i < INSTANCES_SIZE; ++i){
		Labels.emplace_back(unif(re));
	}//Encode the labels to 1 or 0
	for(int i = 0; i < Labels.size(); ++i){
		if (Labels[i] > 0.5) Labels[i] = 1.0;
		else Labels[i] = 0.0;
	}

	unique_ptr<NeuralNet<double>> net( new NeuralNet<double>);
	net->train( Inputs, Labels );
	net->think( Inputs , "test.dat");
	
	//Clean up inputs
	for (auto x : Inputs){
		delete x;
	}
	return(0);
}
