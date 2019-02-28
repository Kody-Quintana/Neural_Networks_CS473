#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <limits>


using namespace std;

template<class T>
class HyperPlaner{
	public:
		HyperPlaner();
		void train( vector<vector<T>> x_instances, vector<T> y_labels );
	private:
		void next_epoch();
		void linear_error();
};


template<class T>
void HyperPlaner<T>::train( vector<vector<T>> x_instances, vector<T> y_labels ){
	if ( x_instances[0].size() != y_labels.size() ) cout << "X and Y must be same size" << endl;
	T b_gradient = 0;
	T m_gradient = 0;




template<class T>
T linear_error( pair<T,T> weights, vector<pair<T,T>> points ){
	T error = 0.0;
	for (long unsigned int i=0; i < points.size(); ++i){
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
	return make_pair( b - (learning_rate * b_gradient),
			m - (learning_rate * m_gradient) );
}



int main( int argc, char** argv ){
	cout.precision(17);
	vector<pair<float,float>> points = {
		/*X    Y*/
		{1.0, 3.0}, 
		{3.0, 2.7}, 
		{7.0, 8.0}, 
		{9.0, 7.6}
	};	
	pair<float,float> W{ 0, 0 };
	float learning_rate = 0.0001;
	cout << linear_error(W, points) << endl;

	float error_so_far = numeric_limits<float>::max();
	float lowest_error = numeric_limits<float>::max();
	pair<float,float> best;
	while( error_so_far > 0.1 ){
		error_so_far = linear_error( W, points );
		W = next_epoch( W, points, learning_rate );
		if (error_so_far < lowest_error){
			lowest_error = error_so_far;
			best = W;
		}
	}
	cout << "M: " << best.second << "  B: " << best.first << endl;

	return(0);
}
