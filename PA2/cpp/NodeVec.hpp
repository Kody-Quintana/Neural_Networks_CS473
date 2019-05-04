#ifndef KNODEVECTOR
#define KNODEVECTOR

#include <iostream>
#include <vector>

//#include "NeuralNet.hpp"

using namespace std;

struct NN_Node{
	const int layer;
	const int l_node;
	vector<NN_Node*> next_paths;
	vector<NN_Node*> prev_paths;

	//Set to 1.0 incase this node gets used as a bias node
	//if it isnt, the value will get overwritten on first f prop.
	double value = 1.0;

	//Reserve space for vectors on construction, set layer and l_node
	//next_paths, and prev_paths are not filled yet
	//if you don't want full connectivity
	NN_Node(int this_layer, int this_node, int next_size, int prev_size);
};

class NodeVector{
	typedef std::pair<int, int> Range;
	friend class NeuralNet;
	private:
		const vector<int> n_per_layer; //Count of nodes per layer, not including the bias nodes
		const int layers; //Number of layers
		const vector<int> n_per_layer_bias; //Count of nodes, inclusing the bias nodes
		const int full_size; //Total number of nodes
		const vector<Range> n_indices; //Ranges for each layer, not including the bias node
		const vector<Range> n_indices_bias; //Ranges for each layer, with the bias nodes
		const int possible_path_size;
		vector<NN_Node> Nodes; //Single vector of all nodes from all layers
		void set_connectivity(); //Hard-coded to full connectivity
	public:
		NodeVector(vector<int> layer_sizes);
};
#endif
