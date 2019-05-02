#include "WeightMat.hpp"
#include "NodeVec.hpp"

using namespace std;

int main( int argc, char** argv ){
	//WeightMatrix test( vector<int>{2, 4, 3, 6} );
	WeightMatrix test({2, 4, 3, 6});

	test.at(2,5,2) = 69.0;
	test.print_all();

	NodeVector n_test({2,3,6,4});


	return(0);
}
