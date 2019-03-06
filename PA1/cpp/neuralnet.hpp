#include <cmath>
#include <vector>
#include <memory>
#include <array>
#include <iostream>
#include <random>

#include "dotproduct.hpp"

using std::vector;
using std::unique_ptr;
using std::cout;
using std::endl;


template<class T>
class NeuralNet{
	public:
	//	NeuralNet();
	//	~NeuralNet();
		void train( vector<vector<T>*> Inputs, vector<T> Labels );
	private:
		vector< vector<T>* > Inputs;
		vector<T> Labels;
		vector<T> Weights;
		T y_hat(int instance);
		T error(int instance);
};



template<class T>
T NeuralNet<T>::y_hat(int instance){
	T y_hat_value = 0.0;
	for (int i = 0; i < Weights.size(); ++i)
		y_hat_value += Weights[i] * (*Inputs[i])[instance];
	return y_hat_value;
}



template<class T>
T NeuralNet<T>::error(int instance){
	return (Labels[instance] - y_hat(instance));
}



template<class T>
void NeuralNet<T>::train( vector<vector<T>* > arg_inputs, vector<T> arg_labels ){
	Inputs = arg_inputs;
	Labels = arg_labels;


	//Initialize Weights with random values
	Weights.reserve(Inputs.size());
	std::uniform_real_distribution<double> unif(0.1, 10.0);
	std::default_random_engine re;
	for (int i = 0; i < Inputs.size(); ++i){
		Weights.emplace_back(unif(re));
	}
}



