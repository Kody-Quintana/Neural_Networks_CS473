#include <iostream>
#include <vector>
#include <cmath>
#include <utility>

using namespace std;



template<class T>
T linear_error( pair<T,T> weights, vector<pair<T,T>> points ){
	T error = 0.0;
	for (int i=0; i < points.size(); ++i){
		T const y = points[i].second;
		T const x = points[i].first;
		T const b = weights.first;
		T const m = weights.second;
		error += pow( y - (m * x + b), 2);
	}
	return error / points.size();
}

template<class T>
pair<T,T> next_epoch( pair<T,T> weights, vector<pair<T,T>> points, T learning_rate ){
	T b_gradient = 0;
	T m_gradient = 0;
	int const N = points.size();
	T const b = weights.first;
	T const m = weights.second;
	for (int i=0; i < N; ++i){
		T const x = points[i].first;
		T const y = points[i].second;
		b_gradient += -(2.0/N) * (y - (m * x) + b);
		m_gradient += -(2.0/N) * x * (y - (m * x) + b);
	}
	cout << "b_gradient: " << b_gradient << endl;
	cout << "m_gradient: " << m_gradient << endl;
	return make_pair( b - (learning_rate * b_gradient),
			m - (learning_rate * m_gradient) );
}



int main( int argc, char** argv ){
	cout.precision(17);
	vector<pair<float,float>> points = {
		{1.0, 3.0}, 
		{3.0, 2.7}, 
		{7.0, 8.0}, 
		{9.0, 7.6}
	};	
	pair<float,float> W{ 0, 0 };
	float learning_rate = 0.00001;
	cout << linear_error(W, points) << endl;

	while( linear_error( W, points ) > 1 ){
		W = next_epoch( W, points, learning_rate );
	}
	cout << "M: " << W.second << "  B: " << W.first << endl;

	return(0);
}
