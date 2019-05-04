#include "WeightMat.hpp"
#include "NodeVec.hpp"
#include "NeuralNet.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>

using std::vector;
using std::cout;
using std::endl;


NeuralNet::NeuralNet(vector<int> layer_sizes) :
	NV(layer_sizes),
	W(NV.n_per_layer_bias),
	error( vector<double>(layer_sizes.back(), 0.0))
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


Gradient::Gradient(int layer, int node, double start_value, vector<NN_Node*>* ready_next) : 
	value(start_value), current_layer(layer), current_node_pos(node), next_paths(ready_next)
{
	//Constructor body
	cout << "Gradient constructed with starting value of: " << value << endl;
}


double NeuralNet::activation(double sum){
	return 1.0 / (1.0 + exp(-sum));
}

double NeuralNet::dx_activation(double activated_sum){
	return activated_sum * (1.0 - activated_sum);
}


void NeuralNet::backward_prop(){
	for (int layer = 0; layer < NV.layers; ++layer){
		for (int node = NV.n_indices_bias[layer].first; node < NV.n_indices_bias[layer].second; ++node){
			for (auto const &nn : NV.Nodes[node].next_paths){
				cout << "    -► Next node at layer: " << nn->layer << ", pos: " << nn->l_node;
				cout << ", through W: " << W.index(nn->layer, nn->l_node, NV.Nodes[node].l_node) << endl;
				vector<Gradient> path_storage;
				path_storage.reserve(NV.possible_path_size);
				path_storage.emplace_back(Gradient(
							nn->layer,
							nn->l_node,
							(NV.Nodes[node].value * dx_activation(nn->value)),
							&nn->next_paths)
							);
				long unsigned int st_index = 0;
				while (st_index < path_storage.size()){
					if (path_storage[st_index].next_paths->size() > 1){
						//When traversing towards the output, here the partial derivative has more then one branch
						//To account for each branch, the current derivative is copied for each possible next step
						//Each copy is assigned one of the possible forward paths
						//During this assignment the copy's gradient is updated with the partial from
						//its current node to its assigned node.
						//This movement has two components:
						// 1) the weight between each node
						// 2) the d/dx of the assigned node's activation
						for (long unsigned int n = 1; n < path_storage[st_index].next_paths->size(); ++n){
							path_storage.emplace_back(path_storage[st_index]);
							path_storage.back().next_paths = &(*path_storage[st_index].next_paths)[n]->next_paths;
							path_storage.back().value *= (
								W(
									(*path_storage[st_index].next_paths)[n]->layer,
									(*path_storage[st_index].next_paths)[n]->l_node,
									path_storage[st_index].current_node_pos
								) * dx_activation( (*path_storage[st_index].next_paths)[n]->value )
							);
							//cout << "L" << 
						}
						Gradient temp = path_storage[st_index];
						temp.next_paths = &(*path_storage[st_index].next_paths)[0]->next_paths;
						temp.value *= (
							W(
								(*path_storage[st_index].next_paths)[0]->layer,
								(*path_storage[st_index].next_paths)[0]->l_node,
								path_storage[st_index].current_node_pos
							) * dx_activation( (*path_storage[st_index].next_paths)[0]->value )
						);
						path_storage[st_index] = temp;
					}
					else if (path_storage[st_index].next_paths->size() == 1){
						//Similar to above, this case handles when there is only one possible path moving forward
						Gradient temp = path_storage[st_index];
						temp.next_paths = &(*path_storage[st_index].next_paths)[0]->next_paths;
						temp.value *= (
							W(
								(*path_storage[st_index].next_paths)[0]->layer,
								(*path_storage[st_index].next_paths)[0]->l_node,
								path_storage[st_index].current_node_pos
							) * dx_activation( (*path_storage[st_index].next_paths)[0]->value )
						);
						path_storage[st_index] = temp;
					}
					else{
						//Finally this branch has reached an output node and there is no more
						//nodes to traverse, here this branch's derivative is updated with 
						//the last d/dx which is the partial of this output node with respect to
						//the total Error
						//Note: at this point only this single branch has completed its traversal,
						//now the path_storage index will be incremented and the next branch will
						//continue traversing until an output node is reached
						//Once all branches have finished they will be summed to create the gradient
						st_index += 1;
						//TODO add the final error d/dx here
						path_storage[st_index].value *= -0.5;
					}
				}
				//End while
				double branch_sum = std::accumulate(begin(path_storage), end(path_storage), 0.0,
						[](double i, const Gradient& grad) {return grad.value + i; }
						);
				cout << "Branch sum: " << std::scientific << branch_sum << endl;
			}
		}
	}
}


void NeuralNet::forward_prop(){
	for (int layer = 1; layer < NV.layers; ++layer){
		//Update all nodes except bias nodes which will always be 1.0
		for (int node = NV.n_indices[layer].first; node < NV.n_indices[layer].second; ++node){
			cout << NV.Nodes[node].value << " -> ";
			NV.Nodes[node].value = 0.0;
			for (const auto p_node : NV.Nodes[node].prev_paths){
				NV.Nodes[node].value += 
					p_node->value * W(NV.Nodes[node].layer, NV.Nodes[node].l_node, p_node->l_node);
			}
			cout << "A(" << NV.Nodes[node].value << ")";
			NV.Nodes[node].value = activation(NV.Nodes[node].value);
			cout << std::fixed << " -> " << NV.Nodes[node].value << endl;
		}
		cout << "\n";
	}
	for (int node = NV.n_indices[NV.layers-1].first; node < NV.n_indices[NV.layers-1].second; ++node){
		cout << "Output: " << NV.Nodes[node].l_node << ", value: " << NV.Nodes[node].value << endl;
		//TODO put into an error vector of (label[i] - prediction[i])
	}
	for (const auto &node : NV.Nodes){ cout << node.value << endl;}
}
