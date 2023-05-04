# Simple and Multiple Layer Perceptron Implementation

This project aims to implement a Simple and Multiple Layer Perceptron using Python 3.10. It includes a `pipfile` that lists all the dependencies required to run the project.

## Dependencies
- Python 3.10

## Usage
There are five run scripts in this project, each of which receives a configuration file as an argument:
- `run_1.py` - runs Simple Perceptron with ex1.json configuration file.
- `run_2.py` - runs Simple Perceptron with ex2.json configuration file.
- `run_3a.py` - runs Multiple Layer Perceptron with ex3a.json configuration file.
- `run_3b.py` - runs Multiple Layer Perceptron with ex3b.json configuration file.
- `run_3c.py` - runs Multiple Layer Perceptron with ex3c.json configuration file.

| Each script corresponds to a specific challenge 

To run any of the scripts, first install the dependencies using pipenv:

```shell
pipenv install
```

Then run the desired script with the appropriate configuration file:

```shell
pipenv run python run_X.py exX.json
```

## Configuration files

Each configuration file is in JSON format and specifies the parameters for the corresponding perceptron. Here is an example of the structure of the configuration file:

```json
{
  "activation_function": {
    "function": "sigmoid",
    "beta": 0.8
  },
  "cut_condition": {
    "condition": "mse",
    "eps": 0.001
  },
  "optimization_method": {
    "method": "gradient",
    "learning_rate": 0.1
  },
  "epochs": 500
}
```


- `activation_function` - an object that specifies the activation function used by the perceptron. It contains:
  - `function` - a string representing the name of the activation function (e.g., "sigmoid", etc.)
  - `beta` - a float representing a parameter of the activation function (e.g., the slope of the sigmoid function)
- `cut_condition` - an object that specifies the stopping criterion for training the perceptron. It contains:
  - `condition` - a string representing the name of the stopping criterion (e.g., "mse" for mean squared error)
  - `eps` - a float representing the threshold value for the stopping criterion
- `optimization_method` - an object that specifies the optimization method used for training the perceptron. It contains:
  - `method` - a string representing the name of the optimization method (e.g., "gradient" for gradient descent)
  - `learning_rate` - a float representing the learning rate used by the optimization method
- `epochs` - an integer representing the number of epochs to train the perceptron for
