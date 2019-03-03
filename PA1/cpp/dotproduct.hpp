#ifndef DOT_PRODUCT_HPP
#define DOT_PRODUCT_HPP

#include <vector>

using std::vector;

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
#endif
