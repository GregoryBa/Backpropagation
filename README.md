# Solving XOR-gate problem
Solving xor-gate problem using backpropagation.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To run you will need:

```
Python 2.7 or higher
Numpy
```

### Installing

To run this program install numpy first

```
pip3 install numpy (linux)
pip install numpy (windows)
```
To run this program:

```
python3 backpropagation.py 
or
python backpropagation
```

## Testing
	- How the inputs are mapped to the output?Conclusion after further analysis of the weights:
The forward pass calculated predicted_output. The first forward pass compared predicted_output with xor_output. Depending on this comparison, the weights for hidden and output layers change using backpropagation. We want to find the weight vector with minimum error (squared error loss), also the minimum point of the error on the gradient graph. The script loops itself until the predicted_output is as close as possible to the xor_output (it gets closer the more epochs we specify in the script). 
I included both the exact and rounded the answer in the script.