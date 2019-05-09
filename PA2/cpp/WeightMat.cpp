#include "WeightMat.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

WeightMatrix::WeightMatrix( std::vector<int> sizes ) :

	layer_sizes( sizes ),

	layers( sizes.size() ),

	//L is used to translate the first arg of WeightMatrix(X,X,X)
	// to the appropriate offset because each set of matrices 
	// is a different size
	L( [&]() -> std::vector<int> {
		std::vector<int> t(layers-1, 0);
		for (int i = 1; i < layers-1; ++i){
			int this_mat_size = layer_sizes[i-1]*layer_sizes[i];
			for (int j = i; j < layers-1; ++j){
				t[j] += this_mat_size;
			};
		};
		//cout << "Weight L offsets:\n";
		//for (auto i : t){ cout << "  " << i << endl;};
		return t; //Initializes const L to this t.
		}()

	),

	//W is all weights stored in a single vector
	W( [&]() -> std::vector<double> {
		srand(time(NULL));
		int full_length = 0;
		for (int i = 1; i < layers; ++i){
			full_length += layer_sizes[i-1]*layer_sizes[i];
		};
		std::vector<double> rw;
		rw.reserve(full_length);
		//Fill with random between 0 and 1
		std::generate_n(std::back_inserter(rw), full_length,
			[]() -> double{ 
				return rand() * (1.0/RAND_MAX); 
			}
		);
		//for (auto i : rw ){ cout << i << endl; };
		return rw;
		}()
	)
	{
		//Constructor body
	}


double& WeightMatrix::operator()(int a, int b, int c){
	//a-1 because forward propigation starts at layer 1
	//and looks "back" to sum nodes
	//WeightMatrix(0,X,X) should never be called
	return W[ L[a-1] + layer_sizes[a-1]*b + c];
}


int WeightMatrix::index(int a, int b, int c){
	return L[a-1] + layer_sizes[a-1]*b + c;
}


void WeightMatrix::print_all(){
	for (auto i : W){cout << i << endl;};
}
