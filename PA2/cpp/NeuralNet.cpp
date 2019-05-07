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


NeuralNet::NeuralNet(vector<int> layer_sizes, vector<double>& label_ref, vector<double>& input_ref, double learn_rate) :
	NV(layer_sizes),
	W(NV.n_per_layer_bias),
	labels(label_ref),
	inputs(input_ref),
	error( vector<double>(layer_sizes.back(),0.0) ),
	instance_label( vector<double>(layer_sizes.back(),0.0) ),
	learning_rate(learn_rate)
{
	//Constructor body
	//cout << "nodes per layer with bias nodes:" << endl;
	//for (auto n : NV.n_per_layer_bias){
	//	cout << "  " << n << endl;
	//}
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
	//cout << "Gradient constructed with starting value of: " << value << endl;
}


void NeuralNet::stage_inputs_and_labels(){
	for (int node = 0; node < NV.n_indices[0].second; ++node){
		NV.Nodes[node].value = inputs[ NV.n_per_layer[0] * instance_index + node ];
	}
	for (int node = 0; node < NV.n_per_layer.back(); ++node){
		instance_label[node] = labels[ NV.n_per_layer.back() * instance_index + node ];
	}

	if ( (instance_index+1) * NV.n_per_layer[0] == inputs.size() ) { instance_index = 0; }
	else { ++instance_index; }
	double total = 0.0;
	for (auto i : error){ total += pow(i,2); }
	total /= error.size();
	rmse = sqrt(total);
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
				//cout << "Starting back prop to node at layer: " << nn->layer << ", pos: " << nn->l_node;
				//cout << ", through W: " << W.index(nn->layer, nn->l_node, NV.Nodes[node].l_node) << endl;
				vector<Gradient> branch_storage;
				branch_storage.reserve(NV.possible_path_size);
				branch_storage.emplace_back(Gradient(
							nn->layer,
							nn->l_node,
							(NV.Nodes[node].value * dx_activation(nn->value)),
							&nn->next_paths)
							);
				long unsigned int bs_index = 0;
				while (bs_index < branch_storage.size()){
					if (branch_storage[bs_index].next_paths->size() > 1){
						//When traversing towards the output, here the partial derivative has more than one branch.
						//To account for each branch, the current derivative is copied for each possible next step
						//Each copy is assigned one of the possible forward paths
						//During this assignment the copy's gradient is updated 
						//with the partial from its current node to its assigned node.
						//This movement has two parts multiplied with the chain rule
						// 1) the weight between each node
						// 2) the d/dx of the assigned node's activation function
						for (long unsigned int n = 1; n < branch_storage[bs_index].next_paths->size(); ++n){
							branch_storage.emplace_back(branch_storage[bs_index]);
							branch_storage.back().next_paths = &(*branch_storage[bs_index].next_paths) [n]->next_paths;
							branch_storage.back().value *= (
								W(
									(*branch_storage[bs_index].next_paths) [n]->layer,
									(*branch_storage[bs_index].next_paths) [n]->l_node,
									branch_storage[bs_index].current_node_pos
								) * dx_activation( (*branch_storage[bs_index].next_paths)[n]->value )
							);
						}
						Gradient temp = branch_storage[bs_index];
						temp.next_paths = &(*branch_storage[bs_index].next_paths) [0]->next_paths;
						temp.value *= (
							W(
								(*branch_storage[bs_index].next_paths) [0]->layer,
								(*branch_storage[bs_index].next_paths) [0]->l_node,
								branch_storage[bs_index].current_node_pos
							) * dx_activation( (*branch_storage[bs_index].next_paths)[0]->value )
						);
						branch_storage[bs_index] = temp;
					}
					else if (branch_storage[bs_index].next_paths->size() == 1){
						//Similar to above, this case handles when there is only one possible path moving forward
						Gradient temp = branch_storage[bs_index];
						temp.next_paths = &(*branch_storage[bs_index].next_paths)[0]->next_paths;
						temp.value *= (
							W(
								(*branch_storage[bs_index].next_paths)[0]->layer,
								(*branch_storage[bs_index].next_paths)[0]->l_node,
								branch_storage[bs_index].current_node_pos
							) * dx_activation( (*branch_storage[bs_index].next_paths)[0]->value )
						);
						branch_storage[bs_index] = temp;
					}
					else{
						//Finally this branch has reached an output node and there is no more
						//nodes to traverse, here this branch's derivative is updated with 
						//the last d/dx which is the partial of this output node with respect to
						//the total Error
						//Note: at this point only this single branch has completed its traversal,
						//now the branch_storage index will be incremented and the next branch will
						//continue traversing until an output node is reached
						//Once all branches have finished they will be summed to create the gradient

						//No need to multply by -1 here, instead just add the gradient, instead of subtracting
						branch_storage[bs_index].value *= error[branch_storage[bs_index].current_node_pos];

						////If you want to only adjust with the first output node.
						//if (branch_storage[bs_index].current_node_pos != 0){
						//	branch_storage[bs_index].value = 0.0;
						//}

						++bs_index;
					}
				}
				//End while
				double branch_sum = std::accumulate(begin(branch_storage), end(branch_storage), 0.0,
						[](double incoming, const Gradient& grad) {return grad.value + incoming; }
						);
				//cout << "Branch sum: " << std::scientific << branch_sum << endl;
				//cout << "  Updating W:" << W.index(nn->layer, nn->l_node, NV.Nodes[node].l_node) << endl;
				//cout << "    from: " << W(nn->layer, nn->l_node, NV.Nodes[node].l_node) << endl;
				W(nn->layer, nn->l_node, NV.Nodes[node].l_node) += (learning_rate * branch_sum);
				//cout << "      to: " << W(nn->layer, nn->l_node, NV.Nodes[node].l_node) << endl;

			}
		}
	}
}


void NeuralNet::forward_prop(){
	for (int layer = 1; layer < NV.layers; ++layer){
		//Update all nodes except bias nodes which will always be 1.0
		for (int node = NV.n_indices[layer].first; node < NV.n_indices[layer].second; ++node){
			//cout << NV.Nodes[node].value << " -> ";
			NV.Nodes[node].value = 0.0;
			for (const auto p_node : NV.Nodes[node].prev_paths){
				NV.Nodes[node].value += 
					p_node->value * W(NV.Nodes[node].layer, NV.Nodes[node].l_node, p_node->l_node);
			}
			//cout << "A(" << NV.Nodes[node].value << ")";
			NV.Nodes[node].value = activation(NV.Nodes[node].value);
			//cout << std::fixed << " -> " << NV.Nodes[node].value << endl;
		}
		//cout << "\n";
	}
	for (int node = NV.n_indices.back().first; node < NV.n_indices.back().second; ++node){
		//cout << "Output: " << NV.Nodes[node].l_node << ", value: " << NV.Nodes[node].value << endl;
		error[ NV.Nodes[node].l_node ] = (/*Label*/instance_label[NV.Nodes[node].l_node]-/*Prediction*/NV.Nodes[node].value);
		//cout << "Error: " << error[ NV.Nodes[node].l_node ] << endl;
	}
	//for (const auto &node : NV.Nodes){ cout << node.value << endl;}
}

void NeuralNet::train(){
	cout << "\n";
	stage_inputs_and_labels();
	forward_prop();
	backward_prop();

	stage_inputs_and_labels();
	forward_prop();
	backward_prop();

	int iteration = 2;
	while (rmse > 0.002){
		stage_inputs_and_labels();
		forward_prop();
		backward_prop();
		cout << "RMSE: " << rmse << " Iteration: " << iteration++ << "\r";
	}
	cout << endl;
	W.print_all();
}

