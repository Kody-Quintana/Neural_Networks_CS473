#include <iostream>
#include <vector>
#include <cmath>

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


//Activation function
template<class T>
bool squash( T input ){
	double result = ( 1 / (1 + exp(-input)) );
	cout << "Result is: " << result << endl;
	if (result >= 0.5) return true;
	else return false;
}


int main( int argc, char** argv ){
	vector<float> X{ 1.5, 3.6, 7.0, 10.0 };
	vector<float> W{ 0.1, 0.3, 0.5, 0.7 };

	using namespace dp;
	cout << W * X << endl;
	cout << squash(W * X) << endl;
	return(0);
}
