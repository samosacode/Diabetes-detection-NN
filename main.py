import numpy as np  
  
#stuff needed   
SIZE = [8,16,16,2]  

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
	T = np.random.uniform(0.2,1.2)
	x = x - np.max(x)
	return np.exp(x - T) / np.sum(np.exp(x - T))  
  
def forward_pass(I):  
	Z1 = W1 @ I + B1  
	A1 = activation(Z1)   
	Z2 = W2 @ A1 + B2  
	A2 = activation(Z2)  
	Z3 = W3 @ A2 + B3  
	A3 = softmax(Z3)  
	return A3 #uhh... 

pregnancies = min(int(input("pregnancies: ")), 15)
glucose = min(int(input("glucose (mg/dL): ")), 120)
blood_pressure = min(int(input("diastolic blood pressure (mm Hg): ")), 100)
skin_thickness = min(int(input("tricep skin fold thickness (mm): ")), 35)
insulin = min(int(input("insulin (μU/mL): ")), 70)
bmi = min(float(input("BMI: ")), 35.0)
pedigree = min(float(input("diabetes pedigree function: ")), 0.9)
age = min(int(input("age: ")), 60)

dataset = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age])

normalisation_vector = data["normalisation_vector"]
dataset /= normalisation_vector[:8]

result = forward_pass(dataset)
print(f"Non-diabetic: {result[0]*100:.1f}%,  Diabetic: {result[1]*100:.1f}%")
  