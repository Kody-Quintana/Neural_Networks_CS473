#include "WeightMat.hpp"
#include "NodeVec.hpp"
#include "NeuralNet.hpp"

#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>

int main( int argc, char** argv ){

	cout << std::fixed << std::setprecision(10);

	const int INSTANCE_SIZE = 1000;
	const int INPUT_SIZE = 3;
	const int LABEL_SIZE = 2;

	//Generate INPUTS
	vector<double> INPUTS;
	srand(time(NULL));
	std::generate_n(std::back_inserter(INPUTS), INSTANCE_SIZE * INPUT_SIZE,
		[=]() -> double{ 
			return rand() * (1.0/RAND_MAX); 
		}
	);

	////Print INPUTS
	//for (long unsigned int i = 0; i < INPUTS.size() - (INPUT_SIZE - 1); i+=INPUT_SIZE){
	//	for (int n = 0; n < INPUT_SIZE; ++n){
	//		cout << INPUTS[i + n] << " ";
	//	}
	//	cout << endl;
	//}

	//Generate LABELS
	vector<double> LABELS(INSTANCE_SIZE * LABEL_SIZE, 0.0);
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator
	std::uniform_int_distribution<> label_distr(0, LABEL_SIZE - 1); // define the range
	for (long unsigned int i = 0; i < LABELS.size() - (LABEL_SIZE - 1); i+=LABEL_SIZE){
		LABELS[i + label_distr(eng)] = 1.0;
	}

	////Print LABELS
	//for (long unsigned int i = 0; i < LABELS.size() - (LABEL_SIZE - 1); i+=LABEL_SIZE){
	//	for (int n = 0; n < LABEL_SIZE; ++n){
	//		cout << LABELS[i + n] << " ";
	//	}
	//	cout << endl;
	//}

	NeuralNet project( {INPUT_SIZE, 4, 3, LABEL_SIZE}, LABELS, INPUTS, 0.00001, INSTANCE_SIZE );
	project.train();

	return(0);
}
