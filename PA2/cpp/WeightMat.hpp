#ifndef KWEIGHTMATRIX
#define KWEIGHTMATRIX
#include <vector>

using std::vector;

class WeightMatrix{
	private:
		const vector<int> layer_sizes;
		const int layers;
		const vector<int> L;
		vector<double> W;
	public:
		WeightMatrix( std::vector<int> sizes );
		double& operator()(int a, int b, int c);
		int index(int a, int b, int c);
		void print_all();
};
#endif
