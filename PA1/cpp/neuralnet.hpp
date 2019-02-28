template<class T>
class NeuralNet{
	public:
		NeuralNet();
		void train();
		T run();
	private:
		T activation();
		T activation_derivative();
};
