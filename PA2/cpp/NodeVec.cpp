#include "NodeVec.hpp"

#include <iostream>
#include <vector>

using namespace std;


NN_Node::NN_Node(int this_layer, int this_node, int next_size, int prev_size) :

	layer(this_layer),

	l_node(this_node),

	next_paths( [=]()->vector<NN_Node*> {
			vector<NN_Node*> nvec;
			nvec.reserve(next_size);
			return nvec; 
			}()
		  ),

	prev_paths( [=]()->vector<NN_Node*> {
			vector<NN_Node*> pvec;
			pvec.reserve(prev_size);
			return pvec;
			}()
		  )
{
	//Constructor body
	cout
		<< "Node in layer: "
		<< this_layer
		<< " at position: "
		<< this_node
		<< " next_p: "
		<< next_size
		<< " prev_p: "
		<< prev_size << endl;
}


NodeVector::NodeVector(vector<int> layer_sizes) :

	n_per_layer( layer_sizes ),

	layers( n_per_layer.size() ),

	n_per_layer_bias( [&]()->vector<int>{
			vector<int> n_per_lb;
			n_per_lb.reserve(layers);
			for (int i = 0; i < layers-1; ++i){
				n_per_lb.emplace_back(n_per_layer[i]+1);
			}
			n_per_lb.emplace_back(n_per_layer.back());
			return n_per_lb;
			}()
		),

	full_size( [&]()->int {
			int fsize = 0;
			for (auto i : layer_sizes){
				fsize += i;
			}
			//To account for the extra bias nodes:
			fsize += (layers - 1);
			return fsize;
			}()
		),
	n_indices( [&]()->vector<Range> {
			vector<Range> n_ind;
			n_ind.reserve(layers);
			int start = 0;
			for (int layer = 0; layer < layers; ++layer){
				n_ind.emplace_back( make_pair(start, start + layer_sizes[layer]-0) );
				start += layer_sizes[layer]+1;
			}
			cout << "n_indices:\n";
			for (auto i : n_ind) { cout << i.first << "-" << i.second << endl;}
			return n_ind;
			}()
		),

	n_indices_bias( [&]()->vector<Range> {
			vector<Range> n_ind_b;
			n_ind_b.reserve(layers);
			int start = 0;
			for (int layer = 0; layer < layers-1; ++layer){
				n_ind_b.emplace_back( make_pair(start, start + layer_sizes[layer]+1) );
				start += layer_sizes[layer]+1;
			}
			//There is no bias node on last layer:
			n_ind_b.emplace_back( make_pair(start, start + layer_sizes[layers-1]) );
			cout << "n_indices_bias:\n";
			for (auto i : n_ind_b) { cout << i.first << "-" << i.second << endl;}
			return n_ind_b;
			}()
		),

	Nodes( [&]()->vector<NN_Node> {
			vector<NN_Node> node_vec;
			node_vec.reserve(full_size);
			for (int layer = 0; layer < layers; ++layer){
				int node_pos = 0;
				for (int n_count = n_indices_bias[layer].first; n_count < n_indices_bias[layer].second; ++n_count){
					node_vec.emplace_back(
						NN_Node({
							layer,
							node_pos++, 
							(layer + 1 < layers) ? n_per_layer_bias[layer+1] : 0,
							(layer - 1 >= 0) ? n_per_layer_bias[layer-1] : 0
						})
					);
				}
			}
			return node_vec;
			}()
	     )
{
	//Constructor body
	set_connectivity();
}


//This has unnessesary complexity but would be the general form if you didn't actually want full connectivity.
void NodeVector::set_connectivity(){
	for (int layer = 0; layer < layers; ++layer){
		for (int node = n_indices_bias[layer].first; node < n_indices_bias[layer].second; ++node){
			if (layer > 0){
				for (int p_node = n_indices_bias[layer-1].first; p_node < n_indices_bias[layer-1].second; ++p_node){
					Nodes[node].prev_paths.push_back( &Nodes[p_node] );
				}
			}
			if (layer < layers-1){
				for (int n_node = n_indices_bias[layer+1].first; n_node < n_indices_bias[layer+1].second; ++n_node){
					Nodes[node].next_paths.push_back( &Nodes[n_node] );
				}
			}
		}
	}
	for (auto const &node : Nodes){
		cout << "\n";
		cout << "Node at layer: " << node.layer << " pos: " << node.l_node;
		cout << " has " << endl;
		for (auto const &nn : node.next_paths){
			cout << "    -► Next node at layer: " << nn->layer << " pos: " << nn->l_node << endl;
		}
		for (auto const &pn : node.prev_paths){
			cout << "   ◄-  Prev node at layer: " << pn->layer << " pos: " << pn->l_node << endl;
		}
	}
}
