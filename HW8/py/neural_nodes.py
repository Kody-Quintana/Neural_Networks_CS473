import numpy as np


class NeuralNet(object):
    def __init__(self, *nodes_per_layer):
        self.n_inputs = nodes_per_layer[0]
        self.n_outputs = nodes_per_layer[len(nodes_per_layer) - 1]

        self.weight_matrix_list = [None] * (len(nodes_per_layer) -1)
        self.node_array_list = [None] * (len(nodes_per_layer))

        #Create n x m matrices for the weights
        for i in range(0, len(nodes_per_layer) - 1):
            self.weight_matrix_list[i] = np.random.rand(
                nodes_per_layer[i + 1], #Rows
                nodes_per_layer[i]) #Columns

        #Create 1 x n arrays for all nodes except inputs
        for i in range(0, len(nodes_per_layer)):
            self.node_array_list[i] = np.zeros((1, nodes_per_layer[i])).T


    def __str__(self):
        nodes = "\nLayers:\n"
        matrices = "\nMatrices\n"

        for i, matrix in enumerate(self.weight_matrix_list):
            matrices += str(i) + ":\n" + str(matrix) + "\n\n"
        for i, node in enumerate(self.node_array_list):
            nodes += str(i) + ":\n" + str(node) + "\n\n"

        return( matrices + nodes )

    def set_inputs(self, X):
        for i in range(len(X)):
            self.node_array_list[0][i] = X[i]

    def forward(self):
        for i in range(len(self.weight_matrix_list)):
            self.node_array_list[i+1] = np.matmul( self.weight_matrix_list[i], self.node_array_list[i] )


TEST = NeuralNet(3, 3, 2, 1)
X = [1.0, 0.7, 0.3]
TEST.set_inputs(X)
print(TEST)
TEST.forward()
print(TEST)
