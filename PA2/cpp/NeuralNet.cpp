#include "WeightMat.hpp"
#include "NodeVec.hpp"
#include "NeuralNet.hpp"

#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

void NeuralNet::forward_prop(){
};

NeuralNet::NeuralNet(vector<int> layer_sizes) :
	NV(layer_sizes),
	W(NV.n_per_layer_bias)
{
	//Constructor body
	cout << "nodes per layer with bias nodes:" << endl;
	for (auto n : NV.n_per_layer_bias){
		cout << "  " << n << endl;
	}
	for (auto const &node : NV.Nodes){
		cout << "\n";
		cout << "Node at layer: " << node.layer << ", pos: " << node.l_node;
		cout << ", has " << endl;
		for (auto const &nn : node.next_paths){
			cout << "    -► Next node at layer: " << nn->layer << ", pos: " << nn->l_node;
			cout << ", through W: " << W.index(nn->layer, nn->l_node, node.l_node) << endl;
		}
		for (auto const &pn : node.prev_paths){
			cout << "   ◄-  Prev node at layer: " << pn->layer << ", pos: " << pn->l_node;
			cout << ", from W: " << W.index(node.layer, node.l_node, pn->l_node) << endl;
		}
	}
}

void NeuralNet::backward_prop(){
}
