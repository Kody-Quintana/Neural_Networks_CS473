#ifndef NEURALNET
#define NEURALNET

#include "NodeVec.hpp"
#include "WeightMat.hpp"

#include <vector>


using std::vector;

struct Gradient{
	double value;
	vector<NN_Node*>* next_path;
	Gradient(int layer, int next_node, WeightMatrix& );
};

class NeuralNet{
	private:
		NodeVector NV;
		WeightMatrix W;
		void forward_prop();
		void backward_prop();

		double activation();
		double dx_activation();
	public:
		NeuralNet(vector<int> layer_sizes);
};
#endif
