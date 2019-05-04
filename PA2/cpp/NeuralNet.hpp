#ifndef NEURALNET
#define NEURALNET

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
	void stage_next(vector<NN_Node*>* next_paths);
};

class NeuralNet{
	private:
		NodeVector NV;
		WeightMatrix W;

		static double activation(double sum);
		static double dx_activation(double activated_sum);
	public:
		NeuralNet(vector<int> layer_sizes);

		void forward_prop();
		void backward_prop();

		vector<double> error;
		double total_error = 0.0;
};
#endif
