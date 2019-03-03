#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "dotproduct.hpp"
#include "neuralnet.hpp"

using namespace dp;

using namespace std;

int main( int argc, char** argv ){

	vector<double> W{ 0.5, 0.3, 0.6, 0.8, 
	                  0.5, 0.3, 0.6, 0.8, 
			  0.4, 1.0, 0.3, 0.9};
	vector<double> X{ 1.3, 1.7, 3.2, 4.5 };

	{
		using dp::operator*;
		cout << W*X << endl;
	}


	unique_ptr<NeuralNet<double>> net( new NeuralNet<double>);
	//NeuralNet<double>* net = new NeuralNet<double>();
	net->train(W, X, 1000000);
	
	return(0);
}
