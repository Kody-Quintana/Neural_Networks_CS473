#include "WeightMat.hpp"
#include "NodeVec.hpp"
#include "NeuralNet.hpp"

#include <iostream>

int main( int argc, char** argv ){

	NeuralNet project({2,3,3,6});
	project.forward_prop();
	project.backward_prop();

	return(0);
}
