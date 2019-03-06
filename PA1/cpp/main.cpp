#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>

#include "dotproduct.hpp"
#include "neuralnet.hpp"

#define INSTANCES_SIZE 15
#define INPUTS_SIZE 10

using namespace dp;
using namespace std;

int main( int argc, char** argv ){



	//Random double generator
	double lower_bound = 0.0;
	double upper_bound = 1.0;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;



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

	//Fill with random doubles, skip first input because its zeros for the bias
	for (int i = 1; i < INPUTS_SIZE + 1; ++i){
		for (int j = 0; j < INSTANCES_SIZE; ++j){
			Inputs[i]->emplace_back(unif(re));
		}
	}


	//Test print
	for (int i = 0; i < Inputs.size(); ++i){
		cout << "\n\nINPUT# " << i << "\n";
		for (int j = 0; j < Inputs[0]->size(); ++j){
			cout << (*Inputs[i])[j] << endl;
		}
	}

	//Generate labels
	vector<double> Labels;
	Labels.reserve(INSTANCES_SIZE);
	for (int i = 0; i < INSTANCES_SIZE; ++i){
		Labels.emplace_back(unif(re));
	}


	//unique_ptr<NeuralNet<double>> net( new NeuralNet<double>);
	//net->train(Inputs, Y, 10000);
	
	//Clean up inputs
	for (auto x : Inputs){
		delete x;
	}
	return(0);
}
