import numpy as np
from numpy import ndarray

from src.perceptron import Perceptron


class MultiLayerPerceptron(Perceptron):

    def train_batch(self, initial_data: ndarray[float], expected: ndarray[float]):
        # #initial_data = NxK, #expected = Nx#output_layer, #layers = Mxm_i, #weights_in_neuron = w_ij

        for epoch in range(self._epochs):
            # Feedforward ("predecir") for each layer.
            feedforward_data = [initial_data]
            feedforward_output = [initial_data]
            for i in range(len(self._layers)):
                feedforward_data.append(
                    np.dot(feedforward_output[i], self._layers[i].neurons)
                )
                feedforward_output.append(
                    self._activation_function.evaluate(
                        feedforward_data[i]
                    )
                )

            # Backpropagation

            # Nos libramos del mu TODO: Check
            delta_W = []
            errors = expected - feedforward_output[-1]  # mu * output_size
            derivatives = self._activation_function.d_evaluate(feedforward_data[-1])  # mu * output_size
            delta_i = [errors * derivatives]  # mu * output_size, deberia ser elemento a elemento
            delta_W.append(np.dot(feedforward_output[-2].T, delta_i) * self._learn_rate)

            for i in reversed(range(len(self._layers) - 1)):
                #        mu * output_size  *   (hidden_size * output_size).T
                error = np.dot(delta_i, self._layers[i + 1].T)  # mu * hidden_size
                derivatives = self._activation_function.d_evaluate(feedforward_data[i + 1])  # mu * hidden_size
                delta_i = [error * derivatives]  # mu * hidden_size
                delta_W.append(np.dot(feedforward_output[i].T, delta_i) * self._learn_rate)

            ## Aplico optimizacion con w +dw ???? # TODO: preguntar en clase

            # delta_W = self._learn_rate * delta_i * feedforward_output[-2] # escalar * (mu * output_size) *  =
            # delta_W = []
            # for i in range(len(delta_i)):  # mu
            #     aux = []
            #     for neuron in feedforward_output[-2][i]:  # hidden_size
            #         aux.append(self._learn_rate * delta_i[i] * neuron)
            #         # lo que esta adentro em deberia quedar de tamaño: hidden_size x output_size
            #     delta_W.append(aux)

            # Si no fuera batch, delta seria de tamaño output_size
            # Si no fuera batch, delta_W sería de tamaño output_size x prev_layer_size
            # Siendo batch, delta es de tamaño mu * output_size
            # Siendo batch, delta_W es de tamaño

            # Expected = [[...],[...],[...]] mu * output_size
            # feedforward_output = layer_count * mu * layer_i_size
            # feedforward_output[-1] = [....] mu * output_size

            # error = y - output   (CHECK)
            # output_error = error * sigmoid(output_layer_input) * (1 - sigmoid(output_layer_input))    (CHECK)
            # hidden_error = np.dot(output_error, output_weights.T) * sigmoid(
            # hidden_layer_input) * (1 - sigmoid(hidden_layer_input))
            #
            #     # Actualizamos los pesos utilizando SGD
            #     output_weights = sgd(output_weights, np.dot(hidden_layer_output.T, output_error), lr)
            #     hidden_weights = sgd(hidden_weights, np.dot(X.T, hidden_error), lr)
