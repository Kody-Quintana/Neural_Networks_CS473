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
		T y_hat_with_act(int instance);
		T error(int instance);
		T W_gradient_with_act(int input);
};



template<class T>
T NeuralNet<T>::y_hat(int instance){
	T y_hat_value = 0.0;
	for (int i = 0; i < Weights.size(); ++i)
		y_hat_value += Weights[i] * (*Inputs[i])[instance];
	return y_hat_value;
}


template<class T>
T NeuralNet<T>::y_hat_with_act(int instance){
	T y_hat_value = 0.0;
	for (int i = 0; i < Weights.size(); ++i)
		y_hat_value += Weights[i] * (*Inputs[i])[instance];
	return 1.0 / (1 + exp( -y_hat_value ));
}


template<class T>
T NeuralNet<T>::W_gradient_with_act(int input){
	T adjustment = 0.0;
	T activated_sum;
	for (int i = 0; i < Labels.size(); ++i){
		activated_sum = y_hat_with_act(i);
		adjustment += (Labels[i] - activated_sum) * (activated_sum * (1.0 - activated_sum)) * (*Inputs[input])[i];
	}
	adjustment *= (-1.0 / Labels.size());
	return adjustment;
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
	cout << std::fixed << W_gradient_with_act(0) << endl;
}



