import numpy as np  
import struct  
import random as rnd  
  
#stuff needed   
SIZE = [8,32,32,2]  

"""
W1 = np.array([[rnd.uniform(-1.0,1.0) for i in range(SIZE[0])] for j in range(SIZE[1])])  
W2 = np.array([[rnd.uniform(-1.0,1.0) for i in range(SIZE[1])] for j in range(SIZE[2])])  
W3 = np.array([[rnd.uniform(-1.0,1.0) for i in range(SIZE[2])] for j in range(SIZE[3])])  
  
B1 = np.array([rnd.uniform(-1.0,1.0) for _ in range(SIZE[1])])
B2 = np.array([rnd.uniform(-1.0,1.0) for _ in range(SIZE[2])])
B3 = np.array([rnd.uniform(-1.0,1.0) for _ in range(SIZE[3])])
"""

data = np.load("diabetes_parameters.npz")

W1 = data["W1"]
W2 = data["W2"]
W3 = data["W3"]

B1 = data["B1"]
B2 = data["B2"]
B3 = data["B3"]

#dataset  

dataset = []

with open('dataset/diabetes.csv', 'r') as f:
   dataset = np.loadtxt('dataset/diabetes.csv', delimiter=',', skiprows=1)
        
dataset = np.array(dataset)

non_zero_cols = [1, 2, 3, 4, 5]
for col in non_zero_cols:
    col_values = dataset[:, col]
    median = np.median(col_values[col_values != 0])
    col_values[col_values == 0] = median

normalisation_vector = [4.0, 90.0, 80.0, 20.0, 85.0, 22.0, 0.25, 25.0, 1.0]
for i in range(len(dataset)):
  dataset[i] /= normalisation_vector
  
#main  

def activation(x):  
	return np.maximum(0,x)  
	  
def softmax(x):  
	x = x - np.max(x)  
	return np.exp(x) / np.sum(np.exp(x))  
  
def forward_pass(I):  
	Z1 = W1 @ I + B1  
	A1 = activation(Z1)   
	Z2 = W2 @ A1 + B2  
	A2 = activation(Z2)  
	Z3 = W3 @ A2 + B3  
	A3 = softmax(Z3)  
	return Z1,A1,Z2,A2,Z3,A3 #uhh...  
  
# train  
def loss(predicted, expected):  
	l = 0.0  
	for i in range(len(expected)):  
		l -= expected[i] * np.log(predicted[i] + 10**-9)  
	return(l)  
  
def activation_derivative(x):  
	y = np.array(x)  
	y[y<=0] = 0  
	y[y>0] = 1  
	return y  
	  
def one_hot(x):  
	y = [0,0]
	match x:
		case 0:
			y = [1,0]
		case 1:
			y = [0,1]
	return y  
  
def backpropagation(I, expected, z1,a1,z2,a2,z3,a3):  
    global W1,W2,W3,B1,B2,B3  
      
    dZ3 = a3 - expected  
    dW3 = np.outer(dZ3, a2)  
    dB3 = dZ3  
      
    dA2 = W3.T @ dZ3  
    dZ2 = dA2 * activation_derivative(z2)  
    dW2 = np.outer(dZ2, a1)  
    dB2 = dZ2  
      
    dA1 = W2.T @ dZ2  
    dZ1 = dA1 * activation_derivative(z1)  
    dW1 = np.outer(dZ1, I)  
    dB1 = dZ1  
      
    W3 -= lr * dW3  
    B3 -= lr * dB3  
    W2 -= lr * dW2  
    B2 -= lr * dB2  
    W1 -= lr * dW1  
    B1 -= lr * dB1  
  
def train_loop(start,end):
	avg_loss = 0.0
	for i in range(start,end):  
		input_layer = dataset[i, :8]
		train_labels = dataset[i][8]
		z1,a1,z2,a2,z3,a3 = forward_pass(input_layer)  
		backpropagation(input_layer,one_hot(train_labels),z1,a1,z2,a2,z3,a3)
		avg_loss += loss(a3,one_hot(train_labels))
		avg_loss /= (i if i > 0 else 1)
		
		if i%10 == 0:  
			
			print(avg_loss)  
			print(i)
	np.savez(
    "diabetes_parameters.npz",
    W1=W1, W2=W2, W3=W3,
    B1=B1, B2=B2, B3=B3
)
	  
lr = 0.01

for i in range(15):
	train_loop(1,768)
	data = np.load("diabetes_parameters.npz")

	W1 = data["W1"]
	W2 = data["W2"]
	W3 = data["W3"]

	B1 = data["B1"]
	B2 = data["B2"]
	B3 = data["B3"]