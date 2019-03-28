import numpy as np


class NeuralNet(object):
    def __init__(self, *nodes_per_layer):
        self.n_inputs = nodes_per_layer[0]
        self.n_outputs = nodes_per_layer[len(nodes_per_layer) - 1]

        self.weight_matrix_list = [None] * len(nodes_per_layer)
        self.node_array_list = [None] * (len(nodes_per_layer))

        #Create n x m matrices for the weights
        for i in range(0, len(nodes_per_layer) - 1):
            self.weight_matrix_list[i] = np.random.rand(
                nodes_per_layer[i + 1], #Rows
                nodes_per_layer[i]) #Columns
            print(self.weight_matrix_list[i])
            #print(nodes_per_layer[i])

        print()

        #Create 1 x n arrays for all nodes except inputs
        for i in range(1, len(nodes_per_layer)):
            self.node_array_list[i] = np.zeros((1, nodes_per_layer[i])).T
            print(self.node_array_list[i])


TEST = NeuralNet(1, 3, 4, 5)
