#ifndef KNEURALNET
#define KNEURALNET

#include "NodeVec.hpp"
#include "WeightMat.hpp"

#include <vector>


using std::vector;

struct Gradient{
	double value;
	int current_layer;
	int current_node_pos;
	vector<NN_Node*>* next_paths;
	Gradient(int layer, int node, double start_value, vector<NN_Node*>* ready_next);
};

class NeuralNet{
	private:
		NodeVector NV;
		WeightMatrix W;

		static double activation(double sum);
		static double dx_activation(double activated_sum);
	public:
		NeuralNet(vector<int> layer_sizes, vector<double>& label_ref, vector<double>& input_ref);

		void forward_prop();
		void backward_prop();

		void stage_inputs();
		void stage_labels();

		vector<double>& labels;
		vector<double>& inputs;
		vector<double> error;
		double total_error = 0.0;
		double learning_rate = 0.01;
};
#endif
