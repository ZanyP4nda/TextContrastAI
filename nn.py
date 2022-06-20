import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pickle

# Filenames
dataset_filename = "train_dataset.pkl"
model_filename = "model.pkl"
log_filename = "train.log"

# Hyperparameters
learning_rate = 0.01
num_epoch = 100

# Model
inputs = [0, 0, 0]
label = 0
cost = 0
epoch_costs = []
w1 = []
b1 = [0, 0, 0]
w2 = []
z1, a1, z2, a2 = 0, 0, 0, 0
cost_2, dc_dw2, cost_1, dc_dw1 = 0, 0, 0, 0

# Activation Functions
def ReLU(x):
	out = []
	for i in x:
		out.append(max(0, i))
	return out

def d_ReLU(x):
	return 1 if x > 0 else 0

def sigmoid(x): 
	return 1 / (1 + math.e ** (-1*x))
def d_sigmoid(x):
	sx = sigmoid(x)
	return sx * (1 - sx)

# Model initialisation
def initialise_weights(num_nodes, num_inputs):
	total_num_weights = num_nodes * num_inputs
	# He Initialization
	std = math.sqrt(2.0 / num_inputs)
	random_weights = list(np.random.randn(total_num_weights) * std)
	# If only have 1 node, return all generated weights
	if num_nodes == 1:
		return random_weights
	# Return weight array for layer
	w = []
	for i in range(0, total_num_weights, num_inputs):
		w.append(random_weights[i:i+num_inputs])
	return w

def initialise_model():
	global w1, w2
	w1 = initialise_weights(3, 3)
	w2 = initialise_weights(1, 3)

def initialise_test_model():
	global w1, w2
	w1 = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.2]]
	w2 = [0.1, 0.3, 0.5]

# Feed forward
def feed_forward():
	global z1, a1, z2, a2
	z1 = np.add(np.matmul(w1, inputs), b1)
	a1 = ReLU(z1)

	z2 = np.matmul(w2, a1)
	a2 = sigmoid(z2)

# Binary Cross Entropy
def get_cost(outputs, batch_labels):
	cost = 0
	for i in range(len(outputs)):
		cost += batch_labels[i] * math.log10(outputs[i]) + (1 - batch_labels[i]) * math.log10(1 - outputs[i])	
	cost *= -1 / len(outputs)
	return cost

def get_d_cost(outputs, batch_labels):
	d_cost = 0
	for i in range(len(outputs)):
		d_cost += (outputs[i] - batch_labels[i]) / (math.log(10) * (outputs[i] - 1) * outputs[i])
	d_cost *= -1 / len(outputs)
	return d_cost

# Backpropagation
def back_propagate(batch_size, d_cost, batch_labels, batch_inputs, z1s, a1s, z2s, a2s):
	global cost_2, dc_dw2, cost_1, dc_dw1
	batch_cost_2, batch_dc_dw2, batch_cost_1, batch_dc_dw1 = [], [], [], []

	for i in range(batch_size): # For every datapoint in batch
		dp_cost_2 = d_cost * d_sigmoid(z2s[i]) # cost_2 for this datapoint
		batch_cost_2.append(dp_cost_2)
		dp_dc_dw2 = dp_cost_2 * np.array(a1s[i]) # dc_dw2 for this datapoint
		batch_dc_dw2.append(dp_dc_dw2)

		dp_cost_1 = []
		dp_dc_dw1 = []
		for j in range(len(w1)): # For each node in h1
			dp_cost_1j = d_ReLU(z1s[i][j]) * dp_cost_2 * w2[j] # Cost of current node
			dp_cost_1.append(dp_cost_1j)

			# Each dc_dw1 is ( cost of current node ) * ( connected input )
			dp_dc_dw1.append([dp_cost_1j * batch_inputs[i][k] for k in range(len(w1))]) 

		batch_cost_1.append(dp_cost_1)
		batch_dc_dw1.append(dp_dc_dw1)

	# Get mean of batch_cost_2
	cost_2 = np.mean(batch_cost_2)
	# Get mean of batch_dc_dw2 (mean of each element in batch_dc_dw2)
	dc_dw2 = np.mean(batch_dc_dw2, axis = 0)
	# Get mean of batch_cost_1 (mean of each element in batch_cost_1)
	cost_1 = np.mean(batch_cost_1, axis = 0)
	# Get mean of batch_dc_dw1 (mean of each element in batch_dc_dw1)
	dc_dw1 = np.mean(batch_dc_dw1, axis = 0)

def update_model():
	global w1, w2, b1
	w2 = np.subtract(w2, (learning_rate * dc_dw2))
	w1 = np.subtract(w1, (learning_rate * dc_dw1))
	b1 = np.subtract(b1, (learning_rate * cost_1))

# Data management
def load_dataset():
	global dataset
	dataset = pickle.load(open(dataset_filename, 'rb'))
	log("Dataset loaded.")

def dump_model():
	model = [w1, w2, b1]
	pickle.dump(model, open(model_filename, 'wb'))
	log("Model dumped.")

# Logging
def open_log():
	global log_file
	log_file = open(log_filename, 'w')
	log("Log opened.")

def close_log():
	log_file.close()

def log(msg):
	log_file.write(msg + "\n")

def feed_forward_log(epoch_num, batch_num):
	log(f"Epoch {epoch_num}, Batch {batch_num}")
	log(f"inputs:\t{inputs}")
	log(f"label:\t{label}")
	log(f"w1:\t{w1}")
	log(f"w2:\t{w2}")
	log(f"b1:\t{b1}")
	log(f"z1:\t{z1}")
	log(f"a1:\t{a1}")
	log(f"z2:\t{z2}")
	log(f"a2:\t{a2}")
	log("="*25)

def back_propagate_log(epoch_num, cost, d_cost):
	global cost_2, dc_dw2, cost_1, dc_dw1
	log(f"Backpropagating: Epoch {epoch_num}")
	log(f"Cost:\t{cost}")
	log(f"d_cost:\t{d_cost}")
	log(f"cost_2:\t{cost_2}")
	log(f"dc_dw2:\t{dc_dw2}")
	log(f"cost_1:\t{cost_1}")
	log(f"dc_dw1:\t{dc_dw1}")
	log("="*25)

def update_model_log():
	log("Updating model")
	log(f"new w1:\t{w1}")
	log(f"new w2:\t{w2}")
	log(f"new b1:\t{b1}")
	log("="*25)
	log("\n")

def train():
	open_log()	
	load_dataset()
	initialise_model()
	for i in tqdm(range(num_epoch)): # Repeat for num_epoch times
		for j in range(len(dataset)): # Repeat for each batch in dataset
			train_batch(dataset[j], i, j)

	dump_model()
	close_log()
	global_min = min(epoch_costs)
	print(f"Global min: {global_min}")
	global_min_epoch = epoch_costs.index(global_min)+1
	print(f"Global min epoch: {global_min_epoch}")
	plt.plot(epoch_costs)
	plt.show()

def train_batch(batch, epoch_num, batch_num):
	global inputs, label, epoch_costs
	batch_labels, z1s, a1s, z2s, a2s = [], [], [], [], []

	batch_inputs = []
	for datapoint in batch:
		# Initialise inputs and label of datapoint
		inputs = datapoint[0]
		batch_inputs.append(inputs)
		label = datapoint[1]
		# Save label to array
		batch_labels.append(label)
		# Feed forward
		feed_forward()
		# Save all calculations to arrays
		z1s.append(z1)
		a1s.append(a1)
		z2s.append(z2)
		a2s.append(a2)
		feed_forward_log(epoch_num, batch_num)

	# Get cost of batch
	cost = get_cost(a2s, batch_labels)
	epoch_costs.append(cost)
	# Get derivative of cost of batch
	d_cost = get_d_cost(a2s, batch_labels)

	# Back propagate
	back_propagate(len(batch), d_cost, batch_labels, batch_inputs, z1s, a1s, z2s, a2s)
	back_propagate_log(epoch_num, cost, d_cost)

	# Update weights
	update_model()
	update_model_log()

train()
