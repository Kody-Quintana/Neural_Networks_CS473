#ifndef KWEIGHTMATRIX
#define KWEIGHTMATRIX
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class WeightMatrix{
	private:
		const vector<int> layer_sizes;
		const int layers;
		const std::vector<int> L;
		std::vector<double> W;
	public:
		WeightMatrix( std::vector<int> sizes );
		double& operator()(int a, int b, int c);
		void print_all();
};
#endif
