#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "dotproduct.hpp"
#include "neuralnet.hpp"

using namespace dp;

using namespace std;

int main( int argc, char** argv ){

	vector<float> W{ 0.5, 0.3, 0.6, 0.8 };
	vector<float> X{ 1.3, 1.7, 3.2, 4.5 };

	cout << dp::operator*(W,X) << endl;


	unique_ptr<NeuralNet<double>> net( new NeuralNet<double>() );
	
	return(0);
}
