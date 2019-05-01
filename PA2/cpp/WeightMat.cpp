#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "WeightMat.hpp"

using namespace std;

WeightMatrix::WeightMatrix( std::vector<int> sizes ) 
		//L is used to translate the first arg of WeightMatrix.at(X,X,X)
		// to the appropriate offset because each set of matrices 
		// is a different size
		: layer_sizes( sizes ),
		layers( sizes.size() ),
		L( [&]() -> std::vector<int> {
			std::vector<int> t(layers, 0);
			for (int i = 1; i < layers; ++i){
				int this_calc = layer_sizes[i-1]*layer_sizes[i];
				for (int j = i; j < layers; ++j){
					t[j] += this_calc;
				};
			};
			//for (auto i : t){ cout << i << endl;};
			return t; //Initializes const L to this t.
			}()
		),//W is all weights stored in a single vector
		W( [&]() -> std::vector<double> {
			int full_length = 0;
			for (int i = 1; i < layers; ++i){
				full_length += layer_sizes[i-1]*layer_sizes[i];
			};
			std::vector<double> rw;
			rw.reserve(full_length);
			//Fill with random between 0 and 1
			std::generate_n(std::back_inserter(rw), full_length,
				[]() -> double{ 
					static double debug = 0.0;
					return debug++;
				//return rand() * (1.0/RAND_MAX); 
				}
			);
			//for (auto i : rw ){ cout << i << endl; };
			return rw;
			}()
		 )
	{
		//Constructor body
	}

double& WeightMatrix::at(int a, int b, int c){
	return W[ L[a] + layer_sizes[a]*b + c];
}

void WeightMatrix::print_all(){
	for (auto i : W){cout << i << endl;};
}
