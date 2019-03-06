#include <cmath>
#include <vector>
#include <memory>
#include <array>
#include <iostream>
#include <random>
#include <chrono>

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
		vector<T> weights_change;
		T y_hat(int instance);
		T y_hat_with_act(int instance);
		T error(int instance);
		T error_with_act(int instance);
		T W_gradient_with_act(int input);
		T weights_rmse();
		T epsilon = 0.00001;
		T learning_rate = 0.001;
};





//Use for prediction
template<class T>
T NeuralNet<T>::y_hat(int instance){
	T y_hat_value = 0.0;
	for (int i = 0; i < Weights.size(); ++i)
		y_hat_value += Weights[i] * (*Inputs[i])[instance];
	return y_hat_value;
}





//Use for classification
template<class T>
T NeuralNet<T>::y_hat_with_act(int instance){
	T y_hat_value = 0.0;
	for (int i = 0; i < Weights.size(); ++i){
		y_hat_value += Weights[i] * (*Inputs[i])[instance];
	}
	return 1.0 / (1.0 + exp( -y_hat_value ));
}



template<class T>
T NeuralNet<T>::error(int instance){
	return (Labels[instance] - y_hat(instance));
}



template<class T>
T NeuralNet<T>::error_with_act(int instance){
	return (Labels[instance] - y_hat_with_act(instance));
}




template<class T>
T NeuralNet<T>::W_gradient_with_act(int input){
	T adjustment = 0.0;
	T activated_sum;
	T N = Labels.size();
	T error_with_act;
	for (int i = 0; i < N; ++i){
		activated_sum = y_hat_with_act(i);
		error_with_act = (Labels[i] - activated_sum);
		adjustment += (-1.0 / N) * (error_with_act) * (activated_sum * (1.0 - activated_sum)) * (*Inputs[input])[i];
	}
	return adjustment;
}









template<class T>
T NeuralNet<T>::weights_rmse(){
	//Compute difference in weights from k to k+1, not sums, for the RMSE
	//The difference of each weight update is stored in std::vector weights_change
	T rmse = 0.0;
	for (T w_diff : weights_change){
		rmse += pow( w_diff, 2 );
	}
	return sqrt( rmse );
}




template<class T>
void NeuralNet<T>::train( vector<vector<T>* > arg_inputs, vector<T> arg_labels ){
	Inputs = arg_inputs;
	Labels = arg_labels;


	//Initialize error of weights with very high number for first iteration
	weights_change.reserve(Inputs.size());
	for (int i = 0; i < Inputs.size(); ++i){
		weights_change.emplace_back( 1.0 );
	}

	//Initialize Weights with random values
	Weights.reserve(Inputs.size());
	std::uniform_real_distribution<double> unif(0.1, 1.0);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	std::default_random_engine re (seed);
	for (int i = 0; i < Inputs.size(); ++i){
		Weights.emplace_back(unif(re));
	}

	T rmse = std::numeric_limits<T>::max();
	while (rmse > epsilon){
	//for (int iteration = 0; iteration < 50; ++iteration){
		rmse = weights_rmse();
		cout << "RMSE: " << std::fixed << rmse << endl;
		//Update weights here
		for (int i = 0; i < Inputs.size(); ++i){

			T W_i_new = Weights[i] - (learning_rate * W_gradient_with_act(i));
			weights_change[i] = Weights[i] - W_i_new;
			Weights[i] = W_i_new;
		}
	}

	cout << "\n";
	for (auto final_weight : Weights){
		cout << final_weight << endl;
	}
}



