#include <cmath>
#include <vector>
#include <memory>
#include <array>
#include <iostream>

using std::vector;
using std::unique_ptr;
using std::cout;
using std::endl;

template<class T>
class NeuralNet{
	public:
		NeuralNet();
		~NeuralNet();
		void train(vector<T> inputs, vector<T> labels, int iterations);
		vector<T> run();
	private:
		T activation(T input);
		T activation_derivative(T input);
		T one = 1.0; //Not sure if this is the right way to specify precision
		vector<T> weights;
		T learning_rate = 0.001;
};


template<class T>
NeuralNet<T>::NeuralNet(){
	cout << "NN constructed" << endl;
}

template<class T>
NeuralNet<T>::~NeuralNet(){
	cout << "NN destructed" << endl;
}


template<class T>
void NeuralNet<T>::train(vector<T> inputs, vector<T> labels, int iterations){

	//2D X matrix is stored in 1D std::vector
	int instances = labels.size(); //Number of x instances should always match the number of y label instances
	int number_of_inputs = inputs.size() / labels.size();
	cout << "Number of inputs: " << number_of_inputs << endl;
	cout << "Number of instances: " << instances << endl;

	//Use of error here is only to adjust weights with higher errors by more than the other weights
	vector<T> error;
	error.reserve( labels.size() );

	//One weight per dimension, plus one bias weight
	weights.reserve( number_of_inputs + 1 );
	//weights( labels.size() + 1, 0.0 );
	for (int i = 0; i < (number_of_inputs + 1); ++i){
		weights.emplace_back( 0.0 );
	}

	//This could also be a while RMSE is greater than something loop
	for (int i = 0; i < iterations; ++i){

		//Iterate through each input
		for (int current_input = 0; current_input < number_of_inputs; ++current_input){

			//Reset gradients between inputs
			T b_gradient = 0.0;
			T m_gradient = 0.0;

			const T m = weights[current_input + 1];
			const T b = weights[0];

			//Iterate through each instance
			for (int instance = 0; instance < labels.size(); ++instance){

				const T this_x = inputs[ (current_input * instances) + instance ];
				const T y = labels[instance];
				
				//Intercept gradient is the derivative of the cost function with respect to b.
				b_gradient += -(2.0 / instances) * (y - (m * this_x) + b);

				//Slope gradient is the derivative of the cost function with respect to m.
				m_gradient += -(2.0 / instances) * this_x * (y - (m * this_x) + b);
			}

			//Adjust bias weight (b)
			weights[0] -= (learning_rate * b_gradient) / number_of_inputs;
			//Divide adjustment by number of inputs so each input has equal influnce over the shared bias

			//Adjust slope weight (m)
			weights[current_input + 1] -= (learning_rate * m_gradient);
		}
	}
	for (auto i : weights){
		cout << i << endl;
	}
}


template <class T>
vector<T> NeuralNet<T>::run(){

}


template <class T>
T NeuralNet<T>::activation(T input){
	return one / (one + exp( -input ));
}


template <class T>
T NeuralNet<T>::activation_derivative(T input){
	return input * (one - input);
}
