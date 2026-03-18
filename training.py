import numpy as np  
import struct  
import random as rnd  
  
#stuff needed   
SIZE = [8,32,32,2]  

W1 = np.random.randn(SIZE[1], SIZE[0]) * np.sqrt(2.0 / SIZE[0])
W2 = np.random.randn(SIZE[2], SIZE[1]) * np.sqrt(2.0 / SIZE[1])
W3 = np.random.randn(SIZE[3], SIZE[2]) * np.sqrt(2.0 / SIZE[2])
B1 = np.zeros(SIZE[1])
B2 = np.zeros(SIZE[2])
B3 = np.zeros(SIZE[3])

try:
	data = np.load("diabetes_parameters.npz")
	
	W1 = data["W1"]
	W2 = data["W2"]
	W3 = data["W3"]
	
	B1 = data["B1"]
	B2 = data["B2"]
	B3 = data["B3"]
except Exception:
	print("file doesn't exist")

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

normalisation_vector = dataset[:, :9].max(axis=0)
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
		avg_loss += loss(a3, one_hot(train_labels))
	
	avg_loss /= (end - start)
	print(avg_loss)
	  
lr = 0.01

for i in range(500):
	np.random.shuffle(dataset)
	train_loop(1,768)
	print(i)

np.savez(
    "diabetes_parameters.npz",
    W1=W1, W2=W2, W3=W3,
    B1=B1, B2=B2, B3=B3, normalisation_vector = normalisation_vector)
	