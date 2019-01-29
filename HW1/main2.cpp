#include <iostream>
#include <vector>

using namespace std;

namespace dp{//Create a namespace to allow use of "Dot Product" operator on std::vector
	template<class T>
	T operator*(vector<T> first, vector<T> second){
		T temp = 0;
		for (int i = 0; i < first.size(); ++i){
			temp += (first[i] * second[i]);
		}
		return( temp );
	}
}

int main( int argc, char** argv ){
	vector<float> W{ 0.5, 0.3, 0.6, 0.8 };
	vector<float> X{ 1.3, 1.7, 3.2, 4.5 };

	using namespace dp;
	cout << W * X << endl;
	return(0);
}
