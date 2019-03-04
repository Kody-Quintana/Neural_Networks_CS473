#include <cmath>
#include <vector>
#include <memory>
#include <array>
#include <iostream>
//#include <ios>

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
		T learning_rate = 0.0001;
		T rmse(vector<T> input);
		vector<T> labels;

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
void NeuralNet<T>::train(vector<T> inputs, vector<T> target_labels, int iterations){
	labels = target_labels;

	//2D X matrix is stored in 1D std::vector
	int instances = labels.size(); //Number of x instances should always match the number of y label instances
	int number_of_inputs = inputs.size() / labels.size();
	cout << "Number of inputs: " << number_of_inputs << endl;
	cout << "Number of instances: " << instances << endl;

	//Normalize labels
	T label_min = labels[0];
	T label_max = labels[0];
	for (int i = 1; i < labels.size() ; ++i){
		label_min = std::min(label_min, labels[i]);
		label_max = std::max(label_max, labels[i]);
	}
	for (int i = 0; i < labels.size() ; ++i){
		labels[i] = (labels[i] - label_min) / (label_max - label_min);
	}

	//1 - 0 encode the y labels
	for (int i = 0; i < labels.size(); ++i){
		if (labels[i] < 0.5) labels[i] = 0.0;
		else labels[i] = 1;
	}
	cout << "Encoded labels: " << endl;
	for (auto i : labels){
		cout << i << endl;
	}

	//Normalize inputs
	T inputs_min = inputs[0];
	T inputs_max = inputs[0];
	for (int i = 1; i < inputs.size() ; ++i){
		inputs_min = std::min(inputs_min, inputs[i]);
		inputs_max = std::max(inputs_max, inputs[i]);
	}
	for (int i = 0; i < inputs.size() ; ++i){
		inputs[i] = (inputs[i] - inputs_min) / (inputs_max - inputs_min);
	}


	//Use of error here is only to adjust weights with higher errors by more than the other weights
	//vector<T> error;
	//error.reserve( labels.size() );

	//One weight per dimension, plus one bias weight
	weights.reserve( number_of_inputs + 1 );
	//weights( labels.size() + 1, 0.0 );
	for (int i = 0; i < (number_of_inputs + 1); ++i){
		weights.emplace_back( 0.0 );
	}

	//This could also be a while RMSE is greater than something loop
	for (int i = 0; i < iterations; ++i){
		T temp_bias_adjustment = 0.0;

		//Iterate through each input
		for (int current_input = 0; current_input < number_of_inputs; ++current_input){

			//Reset gradients at start of new input
			T b_gradient = 0.0;
			T m_gradient = 0.0;

			const T m = weights[current_input + 1];
			const T b = weights[0];

			//Iterate through each instance
			for (int instance = 0; instance < labels.size(); ++instance){

				const T this_x = inputs[ (current_input * instances) + instance ];
				const T y = labels[instance];
				
				//Intercept gradient is the derivative of the cost function with respect to b.
				b_gradient += activation_derivative( -(2.0 / instances) * (y - ( m * this_x + b ) ));

				//Slope gradient is the derivative of the cost function with respect to m.
				m_gradient += activation_derivative(-(2.0 / instances) * (this_x) * (y - ( m * this_x + b) ));

			}//End instances loop

			//Adjust bias weight (b)
			//Divide adjustment by number of inputs so each input has equal influnce over the shared bias
			temp_bias_adjustment -= (learning_rate * b_gradient) / number_of_inputs;
			//Store this adjustment and only update once per epoch (outermost loop)

			//Adjust slope weight (m)
			weights[current_input + 1] -= (learning_rate * m_gradient);

		}//End inputs loop
		weights[0] = temp_bias_adjustment;

	}//End epochs loop
	for (auto i : weights){
		cout << std::fixed << i << endl;
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


template <class T>
T NeuralNet<T>::rmse(vector<T> input){
	T sum = 0.0;
	for (T error : input ){
		sum += pow( error, 2 );
	}
	return sqrt( sum );
}



