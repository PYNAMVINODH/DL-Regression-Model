# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: PYNAM VINODH

### Register Number: 212223240131

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(71)
X=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y = 2 * X + 1 + e

plt.scatter(X,y,c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regresion')
plt.show()

class Model(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear=nn.Linear(in_features,out_features)
    def forward(self,x):
        return self.linear(x)

torch.manual_seed(59)
model=Model(1,1)

initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()
print("\nName: PYNAM VINODH")
print("\nRegister No: 212223240131")
print(f"Initial Weight: {initial_weight:.8f} , Initial Bias: {initial_bias:.8f}\n")

loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

epochs=100
losses=[]


for epoch in range(1,epochs+1):
    optimizer.zero_grad()
    y_pred=model(X)
    loss=loss_function(y_pred,y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

print(f'epoch: {epoch:2} \nloss:{loss.item():10.8f} \nweight: {model.linear.weight.item():10.8f} \nbias: {model.linear.bias.item():10.8f}')

plt.plot(range(epochs),losses,color='coral')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()

final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print("\nName: PYNAM VINODH")
print("\nRegister No: 212223240131")
print(f"Final Weight: {final_weight:.8f} \nFinal Bias: {final_bias:.8f}")

x1=torch.tensor([X.min().item(), X.max().item()])
y1=x1*final_weight+final_bias

plt.scatter(X,y,label="Original Data")
plt.plot(x1,y1,'r',label='Best-Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained model: Best-Fit Line')
plt.legend()
plt.show()

x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new).item()
print("\nName: PYNAM VINODH")
print("\nRegister No: 212223240131")
print(f"Predicted for x=120: {y_new_pred:.8f}")
```

## Dataset Information
<img width="806" height="697" alt="image" src="https://github.com/user-attachments/assets/40d1dadc-699d-4ff8-91f4-847a92f28216" />


## OUTPUT
### Training Loss Vs Iteration Plot
<img width="848" height="679" alt="image" src="https://github.com/user-attachments/assets/d57e9fd3-a8da-4f31-8d72-9e815aa80631" />

<img width="597" height="120" alt="image" src="https://github.com/user-attachments/assets/7d2c6a59-a3c9-4540-91f8-e7343060667b" />



### Best Fit line plot
<img width="727" height="579" alt="image" src="https://github.com/user-attachments/assets/848d432a-e93a-4628-b0af-c9e7132b7f75" />

### New Sample Data Prediction
<img width="419" height="106" alt="image" src="https://github.com/user-attachments/assets/6b4e0ae7-de30-4e25-88d6-7eb84a07577e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
