import numpy as np  
  
#stuff needed   
SIZE = [8,32,32,2]  

data = np.load("diabetes_parameters.npz")

W1 = data["W1"]
W2 = data["W2"]
W3 = data["W3"]

B1 = data["B1"]
B2 = data["B2"]
B3 = data["B3"]

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
	return A3 #uhh... 

pregnancies = int(input("pregnancies: "))
glucose = int(input("glucose (mg/dL): "))
blood_pressure = int(input("diastolic blood pressure (mm Hg): "))
skin_thickness = int(input("tricep skin fold thickness (mm): "))
insulin = int(input("insulin (μU/mL): "))
bmi = float(input("BMI: "))
pedigree = float(input("diabetes pedigree function: "))
age = int(input("age: "))

dataset = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age])

normalisation_vector = np.array([4.0, 90.0, 80.0, 20.0, 85.0, 22.0, 0.25, 25.0])
dataset /= normalisation_vector

result = forward_pass(dataset)
if result[0] < result[1]:
  print("diabetic")
else:
  print("non diabetic")
  