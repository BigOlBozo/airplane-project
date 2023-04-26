# Author: Rett Bull, Pomona College

from typing import List, Tuple, Any

from utilities import *

# Arrays are represented as lists of (equal-sized) lists. Packages like numpy could
# speed up computation if needed


class NeuralNet:
    """A simple implementation of a neural net. Has three layers - input, output and one
    hidden. Contains three lists to hold the activation values for each of the layers
    and four arrays (two for the weights and two more for the most recent changes (for
    momentum))

    Attributes:
        num_input - number of input layer nodes
        num_hidden - number of hidden layer nodes
        num_output - number of output layer nodes
        input_layer - activations (actual values) of input layer neurons
        hidden_layer - activations of hidden layer neurons
        output_layer - activations of output layer neurons
        ih_weights - matrix of weights from input layer to hidden layer (represented as
            nested arrays) Each inner array holds weights mapping all input layer nodes
            to one hidden layer node
        ho_weights - matrix of weights from hidden layer to output layer (represented as
            nested arrays) Each inner array holds weights mapping all input layer nodes
            to one hidden layer node
        ih_weights_changes - changes to ih weights from previous iterations
        ho_weights_changes - changes to ho weights from previous iterations
        act_function_is_sigmoid - whether or not we are currently using sigmoid
        act_function - chosen activation function (defaults to sigmoid)
        dact_function - derivate of activation function (defaults to d_sigmoid), must
            match the activation function
    """

    def __init__(self, n_input: int, n_hidden: int, n_output: int) -> None:
        self.num_input = n_input + 1  # one extra for bias node
        self.num_hidden = n_hidden + 1  # one extra for bias node
        self.num_output = n_output
        self.input_layer: List[float] = [1.0] * self.num_input
        self.hidden_layer: List[float] = [1.0] * self.num_hidden
        self.output_layer: List[float] = [1.0] * self.num_output
        self.ih_weights = make_random_array(self.num_input, self.num_hidden - 1)
        self.ho_weights = make_random_array(self.num_hidden, self.num_output)
        self.ih_weights_changes = make_zero_array(self.num_input, self.num_hidden - 1)
        self.ho_weights_changes = make_zero_array(self.num_hidden, self.num_output)
        self.act_function_is_sigmoid = True
        self.act_function = sigmoid
        self.dact_function = d_sigmoid

    def evaluate(self, inputs: List[Any]) -> List[float]:
        """Carries out forward propagation on the neural net

        Args:
            inputs - list of initial input activations

        Returns:
            output of neural net
        """
        # subtract one for bias
        if len(inputs) != (self.num_input - 1):
            raise SizeMismatch(self.num_input - 1, len(inputs))

        # set input layer + added bias
        self.input_layer = inputs + [1.0]

        # compute input -> hidden activations
        self.hidden_layer = self.compute_one_layer(
            self.input_layer, self.num_hidden, self.ih_weights, True
        )
        # compute hidden -> output activations
        self.output_layer = self.compute_one_layer(
            self.hidden_layer, self.num_output, self.ho_weights, False
        )

        # Return a copy of the output layer.
        return self.output_layer[:]

    def compute_one_layer(
        self,
        curr_layer: List[float],
        num_next_layer: int,
        weights: List[List[float]],
        is_hidden_layer: bool,
    ) -> List[float]:
        """Compute one step of forward propagation (calculate activations of layer x+1
        given activations of layer x and weights from x -> x+1)

        Args:
            curr_layer - activations of current layer
            num_next_layer - number of nodes in next layer
            weights - matrix of weights from current layer to next layer
            is_hidden_layer - whether or not the next layer is a hidden layer, if so
                need to adjust iterations to not affect bias unit of next layer

        Returns:
            computed next layer from current layer and weights
        """
        # 1.0 so that bias is set correctly
        next_layer: List[float] = [1.0] * num_next_layer

        # if hidden layer need to adjust iterations due to bias unit
        iters = num_next_layer - (1 if is_hidden_layer else 0)

        # each outer iteration computes one hidden layer node
        for i in range(iters):
            accum = 0.0
            # each inner iteration adds one weight * one curr layer node to result
            for j in range(len(curr_layer)):
                accum += weights[j][i] * curr_layer[j]

            next_layer[i] = self.act_function(accum)

        return next_layer

    I = List[Any]

    def test(self, data: List[I]) -> List[Tuple[I, List[Any]]]:
        """Tests the neural net on a list of values

        Tricky type signature:

        Takes a list of inputs where each input is a list of ints of floats (using type
        hint of Any as Python typing is still relatively new and has issues with
        restraining types to int or float). The return type is a list of (input, output)
        tuples where output is again a list of ints or float.

        Args:
            data - list of inputs where each input is a list of ints or floats

        Returns:
            list of (input, output) tuples where input is the passed in list while
            output is a list of the neural net's output
        """
        return [(_in, self.evaluate(_in)) for _in in data]

    O = List[Any]

    def test_with_expected(self, data: List[Tuple[I, O]]) -> List[Tuple[I, O, O]]:
        """Tests the neural net on a list of values for which one has ground truth or
        expected results.

        Tricky type signature:

        Takes a list of (input, output) tuples where input and output are each lists
        themselves. These can be lists of ints of floats (using type hint of Any as
        Python typing is still relatively new and has issues with restraining types to
        int or float). The return type is a list of (input, expected output, actual
        output) triples.

        Args:
            data - list of (input, output) tuples where input and output are each lists
                of ints or floats

        Returns:
            list of (input, expected output, actual output) triples where input and
            output are the passed in lists while actual output is a list of the neural
            net's output
        """
        return [(_in, expected, self.evaluate(_in)) for _in, expected in data]

    def train(
        self,
        data: List[Tuple[I, O]],
        learning_rate: float = 0.5,
        momentum_factor: float = 0.1,
        iters: int = 1000,
        print_interval: int = 100,
    ) -> None:
        """Carries out a training cycle on the neural net

        Args:
            data - list of (input, output) tuples where input and output are each lists
                of ints or floats
            learning_rate - scaling factor to apply to derivatives
            momentum_factor - how much influence to give momentum from past updates
            iters - number of iterations to run
            print_interval - how often to print error
        """

        def one_pass():
            """Computes a single backpropagation pass"""
            for x, y in data:
                self.back_propagate(x, y, learning_rate, momentum_factor)

        def one_pass_with_error() -> float:
            """Computes a single backpropagation pass keeping track of error

            Returns:
                error of pass
            """
            error = 0.0
            for (x, y) in data:
                error += self.back_propagate(x, y, learning_rate, momentum_factor)
            return error

        print_count = 0 if (print_interval <= 0) else iters // print_interval
        left_over = iters if (print_interval <= 0) else iters % print_interval

        count = 0
        for i in range(print_count):
            for j in range(print_interval - 1):
                one_pass()
                count += 1
            # one extra iteration of `one_pass_with_error`
            count += 1
            print(f"Error after {count} iterations: {one_pass_with_error()}")

        # finish any remaining passes
        for i in range(left_over):
            one_pass()

    def back_propagate(
        self,
        inputs: List[Any],
        desired_result: List[Any],
        learning_rate: float,
        momentum_factor: float,
    ):
        """The algorithm for adjusting weights

        Computes influence of each node based on derivatives to determine how to adjust
        weights.

        Args:
            inputs - list of input activations (int or float)
            desired_result - expected results
            learning_rate - scaling factor to apply to derivatives
            momentum_factor - how much influence to give momentum from past updates

        Returns:
            error of the pass
        """
        # carry out the forward pass to get actual output
        outputs = self.evaluate(inputs)

        # compute deltas at the output layer
        output_deltas = [
            self.dact_function(out) * (des - out)
            for out, des in zip(outputs, desired_result)
        ]

        # compute deltas at the hidden layer
        hidden_deltas = [0.0] * self.num_hidden
        for h in range(self.num_hidden - 1):
            error = 0.0
            for o in range(self.num_output):
                error += output_deltas[o] * self.ho_weights[h][o]
            hidden_deltas[h] = self.dact_function(self.hidden_layer[h]) * error

        # update weights and changes for hidden -> output layers
        for h in range(self.num_hidden):
            for o in range(self.num_output):
                change = output_deltas[o] * self.hidden_layer[h]
            self.ho_weights[h][o] += (
                learning_rate * change + momentum_factor * self.ho_weights_changes[h][o]
            )
            self.ho_weights_changes[h][o] = change

        # update weights and changes for input -> hidden layers
        for i in range(self.num_input):
            for h in range(self.num_hidden - 1):
                change = hidden_deltas[h] * self.input_layer[i]
                self.ih_weights[i][h] += (
                    learning_rate * change
                    + momentum_factor * self.ih_weights_changes[i][h]
                )
                self.ih_weights_changes[i][h] = change

        # compute square errors (half the sum of squares of the errors)
        square_errors = 0.0
        for o in range(self.num_output):
            square_errors += (desired_result[o] - self.output_layer[o]) ** 2
        return 0.5 * square_errors

    def get_ih_weights(self) -> List[List[float]]:
        """Gets the input-hidden weights as a list of lists

        Returns:
            input layer -> hidden layer weights
        """
        return self.ih_weights

    def get_ho_weights(self) -> List[List[float]]:
        """Gets the input-hidden weights as a list of lists

        Returns:
            hidden layer -> output layer weights
        """
        return self.ho_weights

    def switch_activations(self) -> None:
        """Switches activation function between sigmoid and hyperbolic tangent"""
        self.act_function = tanh if self.act_function_is_sigmoid else sigmoid
        self.dact_function = d_tanh if self.act_function_is_sigmoid else d_sigmoid
