import numpy as np
from numpy import ndarray

from src.perceptron import Perceptron


class MultiLayerPerceptron(Perceptron):

    def test(self, data: ndarray[float]) -> ndarray[float]:
        results = data
        for i in range(len(self._layers)):
            results = np.insert(results, 0, 1, axis=1)
            # results = mu x hidden_size + 1, #layers[i] = (hidden_size + 1) x next_hidden_size
            h = results @ self._layers[i].neurons
            # h = mu x next_hidden_size
            results = self._activation_function.evaluate(h)

        return results

    def train_batch(self, initial_data: ndarray[float], expected: ndarray[float]):
        # #initial_data = mu x initial_size, #expected = mu x output_size
        epoch = 0
        for epoch in range(self._epochs):
            # Feedforward ("predecir") for each layer.
            # Le agrego al initial data los V = 1 para el bias
            feedforward_data = [initial_data]
            results = initial_data
            feedforward_output = []
            for i in range(len(self._layers)):
                results = np.insert(results, 0, 1, axis=1)
                feedforward_output.append(results)
                # results = mu x hidden_size + 1, #layers[i] = (hidden_size + 1) x next_hidden_size
                h = results @ self._layers[i].neurons
                # h = mu x next_hidden_size
                feedforward_data.append(h)
                results = self._activation_function.evaluate(h)

            # Backpropagation using SGD

            # Nos libramos del mu
            delta_W = []
            errors = expected - results  # mu * output_size
            # ver calculo del error con llamando a d_error #

            if self._cut_condition.is_finished(errors):
                return epoch

            derivatives = self._activation_function.d_evaluate(feedforward_data[-1])  # mu * output_size
            delta_i = errors * derivatives  # mu * output_size, elemento a elemento

            # #delta_i = mu * output_size
            # #feedforward_output[-1] = #hidden_data = mu * (hidden_size + 1)
            delta_W.append((feedforward_output[-1].T @ delta_i) * self._learn_rate)
            # #delta_W =  (#hidden_size + 1) * #output_size

            for i in reversed(range(len(self._layers) - 1)):
                # delta_w tiene que tener la suma de todos los delta_w para cada iteracion para ese peso
                #        mu * output_size  *   ((hidden_size + 1 {bias_layer} - 1) * output_size).T
                error = delta_i @ np.delete(self._layers[i + 1].neurons, 0, axis=0).T
                # mu * (hidden_size + 1 {bias_layer} - 1)  == mu * hidden_size

                # Call _optimization_method #
                derivatives = self._activation_function.d_evaluate(feedforward_data[i + 1])  # mu * hidden_size
                delta_i = error * derivatives  # mu * hidden_size
                # #feedforward[i] = mu * (previous_hidden_size + 1) ; delta_i = mu * hidden_size
                delta_W.append((feedforward_output[i].T @ delta_i) * self._learn_rate)
                # Me libero del mu (estoy "sumando" todos los delta_w)

            # Calculo w = w + dw

            for i in range(len(self._layers)):
                self._layers[i].neurons = np.add(self._layers[i].neurons, delta_W[-(i+1)])

        return epoch