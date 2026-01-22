import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model

# Define dataset
xtrain = np.array([[-2.3,-2.1],[-0.6,-2.1],[-3,0.1],[-2.3,1.1],[1.6,-2.7],[0.1,-4.2],
                   [-2.1,-1.4],[-1.6,-0.8],[0,-2.2],[-2.1,0.2],[0,0],[0.7,0.7],[0.6,-0.4],
                   [-0.3,1.6],[-0.3,-0.81],[5.1,-7.9],[2,1.6],[1,1.4],[-1.3,-1.1],[2.3,-0.5],[-0.3,0.7]])
ytrain = np.array([0,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,0,1])

# Define simple model with 2 hidden layers
model = keras.Sequential([
    keras.layers.Dense(3, activation='tanh', input_shape=(2,)),  # First hidden layer with 3 neurons
    keras.layers.Dense(3, activation='tanh', input_shape=(2,))
    keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile and train model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=100, verbose=0)

# Extract first hidden layer outputs
layer_outputs = Model(inputs=model.input, outputs=model.layers[0].output)
hidden_layer_activations = layer_outputs.predict(xtrain)  # Shape: (21, 10)

# Reduce to 2D if needed (taking first 2 neurons for simplicity in this case)
activations_2D = hidden_layer_activations[:, :2]  # Shape: (21, 2)

# 2D Scatter Plot
fig, ax = plt.subplots()

# Scatter points with circles representing different classes
for i in range(len(ytrain)):
    if ytrain[i] == 0:
        ax.scatter(activations_2D[i, 0], activations_2D[i, 1], c='b', marker='o', s=60)  # Class 0: blue circle
    else:
        ax.scatter(activations_2D[i, 0], activations_2D[i, 1], c='r', marker='o', s=60)  # Class 1: red circle

# Add circles as geometric patches
circle1 = plt.Circle((0, 0), 1, fill=False, edgecolor='blue', linestyle='--', linewidth=2)
circle2 = plt.Circle((-2, 0), 1, fill=False, edgecolor='red', linestyle='--', linewidth=2)
circle3 = plt.Circle((0, -2), 1, fill=False, edgecolor='green', linestyle='--', linewidth=2)

# Add circles to the plot
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)

# Labels and title
ax.set_xlabel("Neuron 1 Activation")
ax.set_ylabel("Neuron 2 Activation")
ax.set_title("2D Scatter Plot of First Hidden Layer Activations with Circles")

plt.show()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras
from tensorflow.keras.models import Model

# Define dataset
xtrain = np.array([[-2.3,-2.1],[-0.6,-2.1],[-3,0.1],[-2.3,1.1],[1.6,-2.7],[0.1,-4.2],
                   [-2.1,-1.4],[-1.6,-0.8],[0,-2.2],[-2.1,0.2],[0,0],[0.7,0.7],[0.6,-0.4],
                   [-0.3,1.6],[-0.3,-0.81],[5.1,-7.9],[2,1.6],[1,1.4],[-1.3,-1.1],[2.3,-0.5],[-0.3,0.7]])
ytrain = np.array([0,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,0,1])

# Define the model
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(2,)))
model.add(keras.layers.Dense(3, activation='tanh'))  # First hidden layer
model.add(keras.layers.Dense(3, activation='tanh'))  # Second hidden layer
model.add(keras.layers.Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train the model
model.fit(xtrain, ytrain, epochs=30, batch_size=1)

# Extract the activations from the first hidden layer
layer_outputs = Model(inputs=model.input, outputs=model.layers[0].output)
hidden_layer_activations = layer_outputs.predict(xtrain)  # Shape: (21, 3)

# 3D Scatter Plot based on activations of the first hidden layer
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter points with colors based on class labels
for i in range(len(ytrain)):
    if ytrain[i] == 0:
        ax.scatter(hidden_layer_activations[i, 0], hidden_layer_activations[i, 1], hidden_layer_activations[i, 2], c='b', marker='o', s=60)  # Class 0: blue circle
    else:
        ax.scatter(hidden_layer_activations[i, 0], hidden_layer_activations[i, 1], hidden_layer_activations[i, 2], c='r', marker='o', s=60)  # Class 1: red circle

# Labels and title
ax.set_xlabel("Neuron 1 Activation")
ax.set_ylabel("Neuron 2 Activation")
ax.set_zlabel("Neuron 3 Activation")
ax.set_title("3D Scatter Plot of First Hidden Layer Activations with Class Membership")

plt.show()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras
from tensorflow.keras.models import Model

# Define dataset coordinates and class labels
coords = [[-2.3, -2.1], [-0.6, -2.1], [-3, 0.1], [-2.3, 1.1], [1.6, -2.7], [0.1, -4.2],
          [-2.1, -1.4], [-1.6, -0.8], [0, -2.2], [-2.1, 0.2], [0, 0], [0.7, 0.7], [0.6, -0.4],
          [-0.3, 1.6], [-0.3, -0.81], [5.1, -7.9], [2, 1.6], [1, 1.4], [-1.3, -1.1], [2.3, -0.5], [-0.3, 0.7]]
classes = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

xtrain = np.array(coords)
ytrain = np.array(classes)

# Create the neural network model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2,)),
    keras.layers.Dense(3, activation='tanh'),  # First hidden layer
    keras.layers.Dense(3, activation='tanh'),  # Second hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile and train the model
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=30, batch_size=1)

# Extract the activations of the first hidden layer
layer_outputs = Model(inputs=model.input, outputs=model.layers[0].output)
hidden_layer_activations = layer_outputs.predict(xtrain)  # Shape: (21, 3)

# Create a 3D plot of the first hidden layer's output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points based on class membership, coloring them accordingly
for i in range(len(ytrain)):
    x, y, z = hidden_layer_activations[i, 0], hidden_layer_activations[i, 1], hidden_layer_activations[i, 2]
    if ytrain[i] == 0:
        ax.scatter(x, y, z, c='white', marker='o', s=60, label='Class 0' if i == 0 else "")
    else:
        ax.scatter(x, y, z, c='blue', marker='o', s=60, label='Class 1' if i == 0 else "")

# Add the circles in 3D space
circle_radius = 1  # You can change the circle radius if needed
ax.scatter([0], [0], [0], c='black', label='Circle Centers')  # Add center markers

# Add 3D circles
# Circle centered at (0,0,0)
theta = np.linspace(0, 2 * np.pi, 100)
x_vals = circle_radius * np.cos(theta)
y_vals = circle_radius * np.sin(theta)
z_vals = np.zeros_like(x_vals)  # Flat in the Z-axis

# Plot the circles
ax.plot(x_vals, y_vals, z_vals, c='black', label='Circle: center (0,0,0)')

# Set axis labels and title
ax.set_xlabel('Neuron 1 Activation')
ax.set_ylabel('Neuron 2 Activation')
ax.set_zlabel('Neuron 3 Activation')
ax.set_title('3D Scatter Plot of First Hidden Layer Activations')

# Show the legend
ax.legend()

# Show the plot
plt.show()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras
from tensorflow.keras.models import Model

# Define dataset coordinates and class labels
coords = [[-2.3, -2.1], [-0.6, -2.1], [-3, 0.1], [-2.3, 1.1], [1.6, -2.7], [0.1, -4.2],
          [-2.1, -1.4], [-1.6, -0.8], [0, -2.2], [-2.1, 0.2], [0, 0], [0.7, 0.7], [0.6, -0.4],
          [-0.3, 1.6], [-0.3, -0.81], [5.1, -7.9], [2, 1.6], [1, 1.4], [-1.3, -1.1], [2.3, -0.5], [-0.3, 0.7]]
classes = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

xtrain = np.array(coords)
ytrain = np.array(classes)

# Create the neural network model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2,)),
    keras.layers.Dense(3, activation='tanh'),  # First hidden layer
    keras.layers.Dense(3, activation='tanh'),  # Second hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile and train the model
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=30, batch_size=1)

# Extract the activations of the first hidden layer
layer_outputs = Model(inputs=model.input, outputs=model.layers[0].output)
hidden_layer_activations = layer_outputs.predict(xtrain)  # Shape: (21, 3)

# Create a 3D plot of the first hidden layer's output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points based on class membership, coloring them accordingly
for i in range(len(ytrain)):
    x, y, z = hidden_layer_activations[i, 0], hidden_layer_activations[i, 1], hidden_layer_activations[i, 2]
    if ytrain[i] == 0:
        ax.scatter(x, y, z, c='white', marker='o', s=60, label='Class 0' if i == 0 else "")
    else:
        ax.scatter(x, y, z, c='blue', marker='o', s=60, label='Class 1' if i == 0 else "")

# Add the same 3 circles as in the original 2D plot
circle_radius = 1  # Radius of the circles (can be adjusted if needed)

# Circle 1 centered at (0, 0)
theta = np.linspace(0, 2 * np.pi, 100)
x_vals = circle_radius * np.cos(theta)
y_vals = circle_radius * np.sin(theta)
z_vals = np.zeros_like(x_vals)  # Flat in the Z-axis (for 2D-like behavior)

# Plot the circles in 3D space
ax.plot(x_vals, y_vals, z_vals, c='black', label='Circle 1: Center (0,0)')

# Circle 2 centered at (-2, 0)
x_vals2 = -2 + circle_radius * np.cos(theta)
y_vals2 = circle_radius * np.sin(theta)
ax.plot(x_vals2, y_vals2, z_vals, c='black', label='Circle 2: Center (-2,0)')

# Circle 3 centered at (0, -2)
x_vals3 = circle_radius * np.cos(theta)
y_vals3 = -2 + circle_radius * np.sin(theta)
ax.plot(x_vals3, y_vals3, z_vals, c='black', label='Circle 3: Center (0,-2)')

# Set axis labels and title
ax.set_xlabel('Neuron 1 Activation')
ax.set_ylabel('Neuron 2 Activation')
ax.set_zlabel('Neuron 3 Activation')
ax.set_title('3D Scatter Plot of First Hidden Layer')

# Show the plot
plt.show()

# Create the intermediate model to get the activations of the first hidden layer
intermediate_model = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)  # Layer 1 corresponds to the first hidden layer

# Get activations from the intermediate model
activations = intermediate_model.predict(xtrain)

# Prepare the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the activations in 3D with class membership as color
for i in range(len(ytrain)):
    # For each data point, use activations and color by class
    ax.scatter(activations[i, 0], activations[i, 1], activations[i, 2], 
               c='blue' if ytrain[i] == 1 else 'white', s=60)

ax.set_xlabel('Activation 1')
ax.set_ylabel('Activation 2')
ax.set_zlabel('Activation 3')
ax.set_title('Activations from First Hidden Layer')
plt.show()

# Create the intermediate model to get the activations of the first hidden layer
intermediate_model = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)  # Layer 1 corresponds to the first hidden layer

# Get activations from the intermediate model
activations = intermediate_model.predict(xtrain)

# Create 6 different plots with each axis shared twice
fig = plt.figure(figsize=(15, 12))

# Plot 1: Axis 1, Axis 2, Axis 3
ax1 = fig.add_subplot(231, projection='3d')
for i in range(len(ytrain)):
    ax1.scatter(activations[i, 0], activations[i, 1], activations[i, 2], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax1.set_xlabel('Activation 1')
ax1.set_ylabel('Activation 2')
ax1.set_zlabel('Activation 3')
ax1.set_title('Plot 1 (Axis 1, 2, 3)')

# Plot 2: Axis 1, Axis 3, Axis 2
ax2 = fig.add_subplot(232, projection='3d')
for i in range(len(ytrain)):
    ax2.scatter(activations[i, 0], activations[i, 2], activations[i, 1], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax2.set_xlabel('Activation 1')
ax2.set_ylabel('Activation 3')
ax2.set_zlabel('Activation 2')
ax2.set_title('Plot 2 (Axis 1, 3, 2)')

# Plot 3: Axis 2, Axis 1, Axis 3
ax3 = fig.add_subplot(233, projection='3d')
for i in range(len(ytrain)):
    ax3.scatter(activations[i, 1], activations[i, 0], activations[i, 2], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax3.set_xlabel('Activation 2')
ax3.set_ylabel('Activation 1')
ax3.set_zlabel('Activation 3')
ax3.set_title('Plot 3 (Axis 2, 1, 3)')

# Plot 4: Axis 2, Axis 3, Axis 1
ax4 = fig.add_subplot(234, projection='3d')
for i in range(len(ytrain)):
    ax4.scatter(activations[i, 1], activations[i, 2], activations[i, 0], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax4.set_xlabel('Activation 2')
ax4.set_ylabel('Activation 3')
ax4.set_zlabel('Activation 1')
ax4.set_title('Plot 4 (Axis 2, 3, 1)')

# Plot 5: Axis 3, Axis 1, Axis 2
ax5 = fig.add_subplot(235, projection='3d')
for i in range(len(ytrain)):
    ax5.scatter(activations[i, 2], activations[i, 0], activations[i, 1], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax5.set_xlabel('Activation 3')
ax5.set_ylabel('Activation 1')
ax5.set_zlabel('Activation 2')
ax5.set_title('Plot 5 (Axis 3, 1, 2)')

# Plot 6: Axis 3, Axis 2, Axis 1
ax6 = fig.add_subplot(236, projection='3d')
for i in range(len(ytrain)):
    ax6.scatter(activations[i, 2], activations[i, 1], activations[i, 0], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax6.set_xlabel('Activation 3')
ax6.set_ylabel('Activation 2')
ax6.set_zlabel('Activation 1')
ax6.set_title('Plot 6 (Axis 3, 2, 1)')

plt.tight_layout()
plt.show()

# Create the intermediate model to get the activations of the first hidden layer
intermediate_model = keras.Model(inputs=model.inputs, outputs=model.layers[3].output)  # Layer 1 corresponds to the first hidden layer

# Get activations from the intermediate model
activations = intermediate_model.predict(xtrain)

# Create 6 different plots with each axis shared twice
fig = plt.figure(figsize=(15, 12))

# Plot 1: Axis 1, Axis 2, Axis 3
ax1 = fig.add_subplot(231, projection='3d')
for i in range(len(ytrain)):
    ax1.scatter(activations[i, 0], activations[i, 1], activations[i, 2], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax1.set_xlabel('Activation 1')
ax1.set_ylabel('Activation 2')
ax1.set_zlabel('Activation 3')
ax1.set_title('Plot 1 (Axis 1, 2, 3)')

# Plot 2: Axis 1, Axis 3, Axis 2
ax2 = fig.add_subplot(232, projection='3d')
for i in range(len(ytrain)):
    ax2.scatter(activations[i, 0], activations[i, 2], activations[i, 1], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax2.set_xlabel('Activation 1')
ax2.set_ylabel('Activation 3')
ax2.set_zlabel('Activation 2')
ax2.set_title('Plot 2 (Axis 1, 3, 2)')

# Plot 3: Axis 2, Axis 1, Axis 3
ax3 = fig.add_subplot(233, projection='3d')
for i in range(len(ytrain)):
    ax3.scatter(activations[i, 1], activations[i, 0], activations[i, 2], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax3.set_xlabel('Activation 2')
ax3.set_ylabel('Activation 1')
ax3.set_zlabel('Activation 3')
ax3.set_title('Plot 3 (Axis 2, 1, 3)')

# Plot 4: Axis 2, Axis 3, Axis 1
ax4 = fig.add_subplot(234, projection='3d')
for i in range(len(ytrain)):
    ax4.scatter(activations[i, 1], activations[i, 2], activations[i, 0], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax4.set_xlabel('Activation 2')
ax4.set_ylabel('Activation 3')
ax4.set_zlabel('Activation 1')
ax4.set_title('Plot 4 (Axis 2, 3, 1)')

# Plot 5: Axis 3, Axis 1, Axis 2
ax5 = fig.add_subplot(235, projection='3d')
for i in range(len(ytrain)):
    ax5.scatter(activations[i, 2], activations[i, 0], activations[i, 1], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax5.set_xlabel('Activation 3')
ax5.set_ylabel('Activation 1')
ax5.set_zlabel('Activation 2')
ax5.set_title('Plot 5 (Axis 3, 1, 2)')

# Plot 6: Axis 3, Axis 2, Axis 1
ax6 = fig.add_subplot(236, projection='3d')
for i in range(len(ytrain)):
    ax6.scatter(activations[i, 2], activations[i, 1], activations[i, 0], c='blue' if ytrain[i] == 1 else 'white', s=60)
ax6.set_xlabel('Activation 3')
ax6.set_ylabel('Activation 2')
ax6.set_zlabel('Activation 1')
ax6.set_title('Plot 6 (Axis 3, 2, 1)')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras

datax=[]
datay=[]
coords=[[-2.4,-2],[-0.5,-2.2],[-3.2,0],[-2,1.2],[1.5,-2.8],[0,-4],[-2,-1.3],[-1.8,-0.7],[0,-2],[-2,0],[0,0],[0.8,0.8],[0.7,-0.5],[-0.4,1.5],[-0.33,-0.8],[5,-8],[2,1.5],[1,1.2],[-1.2,-1.1],[2.1,-0.4],[-0.4,0.8]]
for i in range(0,len(coords)):
    datax.append(coords[i][0])
    datay.append(coords[i][1])
xtrain=np.array(coords)
classes=[0,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,0,1]
ytrain=np.array(classes)
trainingdata=tf.data.Dataset.from_tensor_slices(([[-2.4,-2],[-0.5,-2.2],[-3.2,0],[-2,1.2],[1.5,-2.8],[0,-4],[-2,-1.3],[-1.8,-0.7],[0,-2],[-2,0],[0,0],[0.8,0.8],[0.7,-0.5],[-0.4,1.5],[-0.33,-0.8],
[5,-8],[2,1.5],[1,1.2],[-1.2,-1.1],[2.1,-0.4],[-0.4,0.8]],classes))



scat=plt
fig,ax = plt.subplots()
ax.scatter(datax,datay)
ax.add_patch(plt.Circle((0,0),1,fill=False))
ax.add_patch(plt.Circle((-2,0),1,fill=False))
ax.add_patch(plt.Circle((0,-2),1,fill=False))

#plt.show()

#creating the model
model=keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(None,1,2)))
model.add(keras.layers.Dense(3,activation='tanh')) #applies hyperbolic tan
model.add(keras.layers.Dense(3,activation='tanh')) #applies hyperbolic tan
model.add(keras.layers.Dense(1,activation='sigmoid')) #logistic activation

model.summary()

#next need to compile
model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"]) # use this loss for binary classification
#note for presentation: should explain loss, optimizer

#training model

mod=model.fit(xtrain,ytrain,epochs=30,batch_size=1)

#visualize training (see decrease in loss)
plt.clf()
his=mod.history
print(his)
plt.plot(range(0,len(his["loss"])),his["loss"],label="loss")
plt.plot(range(0,len(his["accuracy"])),his["accuracy"],label="accuracy")
plt.legend()
plt.ylim(0,1)
plt.show()



#evaluate
model.evaluate(xtrain,ytrain) #62$ accuracy

model.predict(xtrain) #gives model predictions
model.predict(xtrain).round()
#all 0s

#Note: would be interesting to plot the points along with probabilities to see what model did


#To get output of intermediate layers, not sure if very meaningful without output layer to give prob

extractor=keras.Model(inputs=model.inputs,outputs=[layer.output for layer in model.layers])
features=extractor(xtrain)

inter=keras.Model(inputs=model.inputs,outputs=model.get_layer(index=1).output)
inter_pred=inter.predict(xtrain)



#intermediate models

#first the one layer model
model1=keras.models.Sequential()
model1.add(keras.layers.InputLayer(input_shape=(None,1,2)))
model1.add(keras.layers.Dense(3,activation='tanh')) #applies hyperbolic tan
model1.add(keras.layers.Dense(1,activation='sigmoid')) #logistic activation

model1.summary()

#next need to compile
model1.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])

mod1=model1.fit(xtrain,ytrain,epochs=30,batch_size=1)

model1.predict(xtrain)
model1.predict(xtrain).round()
unique,counts=np.unique(model1.predict(xtrain).round(),return_counts=True)
#15 0s, 6 1s

model1.evaluate(xtrain,ytrain) #52.4% accuracy vs 62% for complicated model

#simplest model
model2=keras.models.Sequential()
model2.add(keras.layers.InputLayer(input_shape=(None,1,2)))
model2.add(keras.layers.Dense(1,activation='sigmoid')) #logistic activation

model2.summary()

#next need to compile
model2.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])

mod2=model2.fit(xtrain,ytrain,epochs=30,batch_size=1)

model2.predict(xtrain)
model2.predict(xtrain).round()
unique,counts=np.unique(model2.predict(xtrain).round(),return_counts=True)
#19 0s, 2 1s

model2.evaluate(xtrain,ytrain) #52.3% accuracy


#addition post presentation (more layers, more neurons)


# 3 layers

model3=keras.models.Sequential()
model3.add(keras.layers.InputLayer(input_shape=(None,1,2)))
model3.add(keras.layers.Dense(3,activation='tanh')) #applies hyperbolic tan
model3.add(keras.layers.Dense(3,activation='tanh'))
model3.add(keras.layers.Dense(3,activation='tanh'))
model3.add(keras.layers.Dense(1,activation='sigmoid')) #logistic activation


# Compile the model
model3.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train the model

mod3=model3.fit(xtrain,ytrain,epochs=30,batch_size=1)

# Plot training history
plt.clf()
his=mod3.history
#print(his)
plt.plot(range(0,len(his["loss"])),his["loss"],label="loss")
plt.plot(range(0,len(his["accuracy"])),his["accuracy"],label="accuracy")
plt.legend()
plt.ylim(0,1)
plt.show()

model3.predict(xtrain)
model3.predict(xtrain).round()
unique,counts=np.unique(model3.predict(xtrain).round(),return_counts=True)
#15 0s, 6 1s

model3.evaluate(xtrain,ytrain) #52.3% for model

# 5 hidden layers


model5=keras.models.Sequential()
model5.add(keras.layers.InputLayer(input_shape=(None,1,2)))
model5.add(keras.layers.Dense(3,activation='tanh')) #applies hyperbolic tan
model5.add(keras.layers.Dense(3,activation='tanh'))
model5.add(keras.layers.Dense(3,activation='tanh'))
model5.add(keras.layers.Dense(3,activation='tanh'))
model5.add(keras.layers.Dense(3,activation='tanh'))
model5.add(keras.layers.Dense(1,activation='sigmoid')) #logistic activation

# Compile the model
model5.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train the model
mod5=model5.fit(xtrain,ytrain,epochs=30,batch_size=1)

# Plot training history
plt.clf()
his=mod5.history
#print(his)
plt.plot(range(0,len(his["loss"])),his["loss"],label="loss")
plt.plot(range(0,len(his["accuracy"])),his["accuracy"],label="accuracy")
plt.legend()
plt.ylim(0,1)
plt.show()


model5.predict(xtrain)
model5.predict(xtrain).round()
unique,counts=np.unique(model5.predict(xtrain).round(),return_counts=True)
#15 0s, 6 1s

model5.evaluate(xtrain,ytrain) # 67%% for complicated model

## 10 layers
model10=keras.models.Sequential()
model10.add(keras.layers.InputLayer(input_shape=(None,1,2)))
model10.add(keras.layers.Dense(3,activation='tanh')) #applies hyperbolic tan
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(3,activation='tanh'))
model10.add(keras.layers.Dense(1,activation='sigmoid')) #logistic activation

# Compile the model
model10.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train the model
mod10=model10.fit(xtrain,ytrain,epochs=30,batch_size=1)

# Plot training history
plt.clf()
his=mod10.history
#print(his)
plt.plot(range(0,len(his["loss"])),his["loss"],label="loss")
plt.plot(range(0,len(his["accuracy"])),his["accuracy"],label="accuracy")
plt.legend()
plt.ylim(0,1)
plt.show()

model10.predict(xtrain)
model10.predict(xtrain).round()
unique,counts=np.unique(model10.predict(xtrain).round(),return_counts=True)
#15 0s, 6 1s

model10.evaluate(xtrain,ytrain) #62% for complicated model


#20 hidden layers

model20=keras.models.Sequential()
model20.add(keras.layers.InputLayer(input_shape=(None,1,2)))
model20.add(keras.layers.Dense(3,activation='tanh')) #applies hyperbolic tan
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(3,activation='tanh'))
model20.add(keras.layers.Dense(1,activation='sigmoid')) #logistic activation

# Compile the model
model20.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train the model
mod20=model20.fit(xtrain,ytrain,epochs=30,batch_size=1)

# Plot training history
plt.clf()
his=mod20.history
#print(his)
plt.plot(range(0,len(his["loss"])),his["loss"],label="loss")
plt.plot(range(0,len(his["accuracy"])),his["accuracy"],label="accuracy")
plt.legend()
plt.ylim(0,1)
plt.show()

model20.predict(xtrain)
model20.predict(xtrain).round()
unique,counts=np.unique(model20.predict(xtrain).round(),return_counts=True)
#15 0s, 6 1s

model20.evaluate(xtrain,ytrain) #62% for complicated model

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define dataset
coords = [[-2.4,-2],[-0.5,-2.2],[-3.2,0],[-2,1.2],[1.5,-2.8],[0,-4],[-2,-1.3],[-1.8,-0.7],[0,-2],[-2,0],
          [0,0],[0.8,0.8],[0.7,-0.5],[-0.4,1.5],[-0.33,-0.8],[5,-8],[2,1.5],[1,1.2],[-1.2,-1.1],[2.1,-0.4],[-0.4,0.8]]
classes = [0,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,0,1]

xtrain = np.array(coords)
ytrain = np.array(classes)

# Define neural network model with 3 hidden layers
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(2,)),
    keras.layers.Dense(3, activation='tanh'),
    keras.layers.Dense(6, activation='tanh'),
    keras.layers.Dense(3, activation='tanh'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train model
mod = model.fit(xtrain, ytrain, epochs=30, batch_size=1, verbose=0)

# Function to plot 3D activations
def plot_activations(layer_output, title, axis_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(layer_output[:, 0], layer_output[:, 1], layer_output[:, 2], c=ytrain, cmap='coolwarm')
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)
    plt.show()

# Extract activations
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
features = extractor(xtrain)

# Plot first hidden layer activations
plot_activations(features[0], 'First Hidden Layer Activations', ['Neuron 1', 'Neuron 2', 'Neuron 3'])

# Plot second hidden layer activations
plot_activations(features[1], 'Second Hidden Layer Activations', ['Neuron 1', 'Neuron 2', 'Neuron 3'])

# Evaluate model
accuracy = model.evaluate(xtrain, ytrain, verbose=0)[1]
print(f'Model Accuracy: {accuracy * 100:.2f}%')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D

def inside_circle(x, y, center, radius):
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

# Generate Data
coords = np.array([[-2.4,-2],[-0.5,-2.2],[-3.2,0],[-2,1.2],[1.5,-2.8],[0,-4],
                    [-2,-1.3],[-1.8,-0.7],[0,-2],[-2,0],[0,0],[0.8,0.8],[0.7,-0.5],
                    [-0.4,1.5],[-0.33,-0.8],[5,-8],[2,1.5],[1,1.2],[-1.2,-1.1],[2.1,-0.4],[-0.4,0.8]])
labels = np.array([0,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,0,1])

# Extend to 3D by adding a zero z-coordinate
coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

# Check which points are inside the unit circles at (0,0) and (-2,0)
inside = np.array([inside_circle(x, y, (0, 0), 1) or inside_circle(x, y, (-2, 0), 1) for x, y, _ in coords])
colors = np.where(inside, 'white', 'blue')

# Create 6 subplots with different axis combinations
fig, axes = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(15, 10))
axis_pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]  # (x,y), (x,z), (y,z), etc.
axis_labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z'), ('Y', 'X'), ('Z', 'X'), ('Z', 'Y')]

for ax, (i, j), (xlabel, ylabel) in zip(axes.flatten(), axis_pairs, axis_labels):
    ax.scatter(coords[:, i], coords[:, j], c=colors, edgecolors='black', s=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{xlabel} vs {ylabel}')
    ax.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Define dataset (fixed formatting)
coords = np.array([
    [-2.4, -2], [-0.5, -2.2], [-3.2, 0], [-2, 1.2], [1.5, -2.8], [0, -4], [-2, -1.3], 
    [-1.8, -0.7], [0, -2], [-2, 0], [0, 0], [0.8, 0.8], [0.7, -0.5], [-0.4, 1.5], 
    [-0.33, -0.8], [5, -8], [2, 1.5], [1, 1.2], [-1.2, -1.1], [2.1, -0.4], [-0.4, 0.8]
])
classes = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

# Scatter plot of data points
fig, ax = plt.subplots()
ax.scatter(coords[:, 0], coords[:, 1], c=['blue' if y == 1 else 'white' for y in classes], edgecolors='black')
ax.add_patch(plt.Circle((0, 0), 1, fill=False))
ax.add_patch(plt.Circle((-2, 0), 1, fill=False))
ax.add_patch(plt.Circle((0, -2), 1, fill=False))
plt.show()

# Creating the neural network model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2,)),  # Fix input shape to (2,)
    keras.layers.Dense(3, activation='tanh'),
    keras.layers.Dense(3, activation='tanh'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Compile model
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train model
history = model.fit(coords, classes, epochs=30, batch_size=1, verbose=1)

# Visualize loss and accuracy over epochs
plt.figure()
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["accuracy"], label="Accuracy")
plt.legend()
plt.ylim(0, 1)
plt.title("Training Loss & Accuracy")
plt.show()

# Evaluate model
model.evaluate(coords, classes)

# Predict class probabilities
predictions = model.predict(coords)
print(predictions.round())  # Rounded predictions

# Extract activations from the first hidden layer
intermediate_model = keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
activations = intermediate_model.predict(coords)

# Create six 3D scatter plots with different axis combinations
fig = plt.figure(figsize=(15, 12))
axis_permutations = [
    (0, 1, 2), (0, 2, 1),
    (1, 0, 2), (1, 2, 0),
    (2, 0, 1), (2, 1, 0)
]
titles = [
    "Plot 1 (Axis 1, 2, 3)", "Plot 2 (Axis 1, 3, 2)",
    "Plot 3 (Axis 2, 1, 3)", "Plot 4 (Axis 2, 3, 1)",
    "Plot 5 (Axis 3, 1, 2)", "Plot 6 (Axis 3, 2, 1)"
]

for i, (x_idx, y_idx, z_idx) in enumerate(axis_permutations):
    ax = fig.add_subplot(231 + i, projection='3d')
    for j in range(len(classes)):
        ax.scatter(
            activations[j, x_idx], activations[j, y_idx], activations[j, z_idx],
            c='blue' if classes[j] == 1 else 'white', edgecolors='black', s=60
        )
    ax.set_xlabel(f'Activation {x_idx+1}')
    ax.set_ylabel(f'Activation {y_idx+1}')
    ax.set_zlabel(f'Activation {z_idx+1}')
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Define dataset
coords = np.array([
    [-2.4, -2], [-0.5, -2.2], [-3.2, 0], [-2, 1.2], [1.5, -2.8], [0, -4], [-2, -1.3], 
    [-1.8, -0.7], [0, -2], [-2, 0], [0, 0], [0.8, 0.8], [0.7, -0.5], [-0.4, 1.5], 
    [-0.33, -0.8], [5, -8], [2, 1.5], [1, 1.2], [-1.2, -1.1], [2.1, -0.4], [-0.4, 0.8]
])
classes = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

# Creating the neural network model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2,)),  
    keras.layers.Dense(3, activation='tanh', name='hidden_layer_1'),
    keras.layers.Dense(3, activation='tanh', name='hidden_layer_2'),
    keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])

model.summary()

# Compile model
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train model
history = model.fit(coords, classes, epochs=30, batch_size=1, verbose=1)

# Extract activations from first and second hidden layers
layer_outputs = [model.get_layer(name="hidden_layer_1").output, model.get_layer(name="hidden_layer_2").output]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

activations_1, activations_2 = activation_model.predict(coords)

# Define axis permutations for 3D plots
axis_permutations = [
    (0, 1, 2), (0, 2, 1),
    (1, 0, 2), (1, 2, 0),
    (2, 0, 1), (2, 1, 0)
]

# Titles for subplots
titles = [
    "Plot 1 (Axis 1, 2, 3)", "Plot 2 (Axis 1, 3, 2)",
    "Plot 3 (Axis 2, 1, 3)", "Plot 4 (Axis 2, 3, 1)",
    "Plot 5 (Axis 3, 1, 2)", "Plot 6 (Axis 3, 2, 1)"
]

# Function to plot activations in 3D
def plot_activations(activations, layer_name):
    fig = plt.figure(figsize=(15, 12))
    for i, (x_idx, y_idx, z_idx) in enumerate(axis_permutations):
        ax = fig.add_subplot(231 + i, projection='3d')
        for j in range(len(classes)):
            ax.scatter(
                activations[j, x_idx], activations[j, y_idx], activations[j, z_idx],
                c='blue' if classes[j] == 1 else 'white', edgecolors='black', s=60
            )
        ax.set_xlabel(f'Activation {x_idx+1}')
        ax.set_ylabel(f'Activation {y_idx+1}')
        ax.set_zlabel(f'Activation {z_idx+1}')
        ax.set_title(f'{layer_name} - {titles[i]}')
    plt.tight_layout()
    plt.show()

# Plot first hidden layer activations
plot_activations(activations_1, "First Hidden Layer")

# Plot second hidden layer activations
plot_activations(activations_2, "Second Hidden Layer")

plot_activations(activations_1, "First Hidden Layer")

plot_activations(activations_2, "Second Hidden Layer")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Define dataset
coords = np.array([
    [-2.4, -2], [-0.5, -2.2], [-3.2, 0], [-2, 1.2], [1.5, -2.8], [0, -4], [-2, -1.3], 
    [-1.8, -0.7], [0, -2], [-2, 0], [0, 0], [0.8, 0.8], [0.7, -0.5], [-0.4, 1.5], 
    [-0.33, -0.8], [5, -8], [2, 1.5], [1, 1.2], [-1.2, -1.1], [2.1, -0.4], [-0.4, 0.8]
])
classes = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

# Create neural network with 10 hidden layers
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(2,)))  

# Add 10 hidden layers
for i in range(1, 11):
    model.add(keras.layers.Dense(3, activation='tanh', name=f'hidden_layer_{i}'))

# Output layer
model.add(keras.layers.Dense(1, activation='sigmoid', name='output_layer'))

model.summary()

# Compile model
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train model
history = model.fit(coords, classes, epochs=30, batch_size=1, verbose=1)

# Extract activations from all 10 hidden layers
layer_outputs = [model.get_layer(name=f"hidden_layer_{i}").output for i in range(1, 11)]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(coords)

# Define axis permutations for 3D plots
axis_permutations = [
    (0, 1, 2), (0, 2, 1),
    (1, 0, 2), (1, 2, 0),
    (2, 0, 1), (2, 1, 0)
]

# Function to plot activations in 3D
def plot_activations(activations, layer_name):
    fig = plt.figure(figsize=(15, 12))
    for i, (x_idx, y_idx, z_idx) in enumerate(axis_permutations):
        ax = fig.add_subplot(231 + i, projection='3d')
        for j in range(len(classes)):
            ax.scatter(
                activations[j, x_idx], activations[j, y_idx], activations[j, z_idx],
                c='blue' if classes[j] == 1 else 'white', edgecolors='black', s=60
            )
        ax.set_xlabel(f'Activation {x_idx+1}')
        ax.set_ylabel(f'Activation {y_idx+1}')
        ax.set_zlabel(f'Activation {z_idx+1}')
        ax.set_title(f'{layer_name} - Plot {i+1}')
    plt.tight_layout()
    plt.show()

# Plot activations for each hidden layer
for i in range(10):
    plot_activations(activations[i], f"Hidden Layer {i+1}")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Define dataset
coords = np.array([
    [-2.4, -2], [-0.5, -2.2], [-3.2, 0], [-2, 1.2], [1.5, -2.8], [0, -4], [-2, -1.3], 
    [-1.8, -0.7], [0, -2], [-2, 0], [0, 0], [0.8, 0.8], [0.7, -0.5], [-0.4, 1.5], 
    [-0.33, -0.8], [5, -8], [2, 1.5], [1, 1.2], [-1.2, -1.1], [2.1, -0.4], [-0.4, 0.8]
])
classes = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

# Create neural network with 10 hidden layers
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(2,)))  

# Add 10 hidden layers
for i in range(1, 11):
    model.add(keras.layers.Dense(3, activation='tanh', name=f'hidden_layer_{i}'))

# Output layer
model.add(keras.layers.Dense(1, activation='sigmoid', name='output_layer'))

model.summary()

# Compile model
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train model
history = model.fit(coords, classes, epochs=30, batch_size=1, verbose=1)

# Extract activations from all 10 hidden layers
layer_outputs = [model.get_layer(name=f"hidden_layer_{i}").output for i in range(1, 11)]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(coords)

# Define axis permutations for 3D plots
axis_permutations = [
    (0, 1, 2), (0, 2, 1),
    (1, 0, 2), (1, 2, 0),
    (2, 0, 1), (2, 1, 0)
]

# Function to plot activations in 3D
def plot_activations(activations, layer_name):
    fig = plt.figure(figsize=(15, 12))
    for i, (x_idx, y_idx, z_idx) in enumerate(axis_permutations):
        ax = fig.add_subplot(231 + i, projection='3d')
        for j in range(len(classes)):
            ax.scatter(
                activations[j, x_idx], activations[j, y_idx], activations[j, z_idx],
                c='blue' if classes[j] == 1 else 'white', edgecolors='black', s=60
            )
        ax.set_xlabel(f'Activation {x_idx+1}')
        ax.set_ylabel(f'Activation {y_idx+1}')
        ax.set_zlabel(f'Activation {z_idx+1}')
        ax.set_title(f'{layer_name} - Plot {i+1}')
    plt.tight_layout()
    plt.show()

# Plot activations for each hidden layer
for i in range(10):
    plot_activations(activations[i], f"Hidden Layer {i+1}")

plot_activations(activations, "Hidden Layer 10")
