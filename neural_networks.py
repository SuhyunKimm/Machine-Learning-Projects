import gzip
import csv
import numpy as np
import sys

def sigmoid(num) :
    return 1 / (1+ np.exp(-num))

def normalize(num) :
    return float(num) / 255

def divide(arr) :
    return arr / mini_batch_size

def FeedForward(sample, weight_arr_1, weight_arr_2, bias_1, bias_2) :
    hidden_layer = np.dot(sample,weight_arr_1) + bias_1 # 1 row 30 cols
    squashed_hidden_layer = sigmoid(hidden_layer)
    
    output_layer = np.dot(squashed_hidden_layer,weight_arr_2) + bias_2 # 1 row 10 cols
    squashed_output_layer = sigmoid(output_layer)
    
    return squashed_hidden_layer, squashed_output_layer

def test(test_set,test_size, test_set_label, w1,w2,b1,b2) :
    accurate_cnt = 0
    for i in range(test_size) :
        normalize_function = np.vectorize(normalize)
        current_row = np.array(normalize_function(test_set[i]))
        hidden, output = FeedForward(current_row,w1,w2,b1,b2)
        if np.argmax(output) == int(test_set_label[i]) :
            accurate_cnt += 1
    accuracy = accurate_cnt / test_size
    return accuracy

def BackPropagation(samples, weight_arr_2, squash_hidden, squash_output, learning_rate, desired_arr, hidden_layer_size, output_layer_size) :
    
    # find output error terms for each neuron in the output layer
    output_delta = (squash_output * (1-squash_output)) * (desired_arr - squash_output)
    output_delta = np.array(output_delta).reshape(1,output_layer_size)
    # update weights from hidden layer to output layer
    
    new_output_delta = np.dot(np.array(squash_hidden).reshape(hidden_layer_size,1), output_delta)
    
    new_weight_arr2 = weight_arr_2 + learning_rate * new_output_delta
    # find bias delta from hidden layer to output layer
    bias_2_delta = np.sum(output_delta, axis=0)
    
    # find hidden error terms for each neuron in the hidden layer
    hidden_delta = (squash_hidden * (1-squash_hidden)) * np.dot(output_delta, new_weight_arr2.T)
    new_hidden_delta = np.dot(np.array(samples).reshape(784,1), hidden_delta)
    # update biases from input layer to hidden layer
    bias_1_delta = np.sum(hidden_delta, axis=0)
    
    return new_output_delta, bias_1_delta, new_hidden_delta, bias_2_delta
    #return new_weight_arr1,new_weight_arr2,new_bias_1,new_bias_2

def NeuralNetworks(train_set, train_size, epoch, mini_batch_size, learning_rate, input_layer_size, hidden_layer_size, output_layer_size, train_set_label) :
    # initializing layer metrics
    weight_arr_1 = np.random.uniform(-1, 1, (input_layer_size, hidden_layer_size)) # weight from input layer to hidden layer
    weight_arr_2 = np.random.uniform(-1, 1, (hidden_layer_size, output_layer_size)) # weight from hidden layer to output layer
    bias_1 = np.random.uniform(-1, 1, hidden_layer_size) # bias for hidden layer
    bias_2 = np.random.uniform(-1, 1, output_layer_size) # bias for output layer
    
    # iteration as many as size of epoch
    iteration = train_size // mini_batch_size
    
    accuracy_arr = []
    
    for i in range(epoch) :
        for j in range(iteration) :
            current_sample = np.array(train_set[mini_batch_size * j : mini_batch_size * (j+1)]) # extracting training sample for the iteration
            current_label = np.array(train_set_label[mini_batch_size * j : mini_batch_size * (j+1)],dtype=int)
            
            total_output_delta = np.zeros(weight_arr_2.shape)
            total_hidden_delta = np.zeros(weight_arr_1.shape)
            total_bias_1_delta = np.zeros([1,hidden_layer_size])
            total_bias_2_delta = np.zeros([1,output_layer_size])
            
            for k in range(len(current_sample)) : # 1D array of pixel values (str type) (1 row, 784 cols)
                normalize_function = np.vectorize(normalize)
                current_row = np.array(normalize_function(current_sample[k]))

                squash_hidden, squash_output = FeedForward(current_row,weight_arr_1,weight_arr_2,bias_1,bias_2)

                target_output = current_label[k]

                desired_arr = np.zeros(output_layer_size)
                
                desired_arr[int(target_output)] = 1
                
                output_delta,bias_1_delta, hidden_delta, bias_2_delta= BackPropagation(current_row, weight_arr_2, squash_hidden,squash_output, learning_rate,desired_arr, hidden_layer_size, output_layer_size)
                total_output_delta += output_delta
                total_hidden_delta += hidden_delta
                total_bias_1_delta += bias_1_delta
                total_bias_2_delta += bias_2_delta
                
            weight_arr_2 = weight_arr_2 + learning_rate * divide(total_output_delta)
            weight_arr_1 = weight_arr_1 + learning_rate * divide(total_hidden_delta)
            bias_1 = bias_1 + learning_rate * divide(total_bias_1_delta)
            bias_2 = bias_2 + learning_rate * divide(total_bias_2_delta)
        accuracy_arr.append(test(test_set,test_size,test_set_label, weight_arr_1,weight_arr_2,bias_1,bias_2))
    print('accuracy array per epoch : ',accuracy_arr)
    print('maximum accuracy : ',max(accuracy_arr))
    return
train_set = []
test_set = []
train_set_label = []
test_set_label = []

train_file_name = 'fashion-mnist_train.csv.gz'
test_file_name = 'fashion-mnist_test.csv.gz'
# # allocation of value by command line
# input_layer_size = sys.argv[1]
# hidden_layer_size = sys.argv[2]
# output_layer_size = sys.argv[3]
# train_file_name = sys.argv[4]
# test_file_name = sys.argv[5]

# open and read train data
with gzip.open(train_file_name, 'rt', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_set_label.append(row[0])
        train_set.append(row[1:])
        

# open and read test data
with gzip.open(test_file_name, 'rt', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_set_label.append(row[0])
        test_set.append(row[1:])

# remove header row from the data arrays        
train_set_header = train_set.pop(0)
test_set_header = test_set.pop(0)
test_set_label.pop(0)
train_set_label.pop(0)

train_size = len(train_set)
test_size = len(test_set)

# epoch, mini-batch size, and learning rate
epoch = 30
mini_batch_size = 20
learning_rate = 3
learning_rate_arr = [0.001,0.01,1.0,10,100]
mini_batch_size_arr = [1,5,20,100,300]
epoch_arr = [10,50,100]
input_layer_size = 784
hidden_layer_size = 30
output_layer_size = 10

print('epoch = ',epoch, ', mini batch size = ', mini_batch_size, ', learning rate = ',learning_rate)
NeuralNetworks(train_set, train_size, epoch, mini_batch_size, learning_rate, input_layer_size, hidden_layer_size, output_layer_size, train_set_label)
print('-------------------------------------\n')

print('NEURAL NETWORKS WITH DIFFERENT LEARNING RATE\n')
for lr in learning_rate_arr :
    print('epoch = ',epoch, ', mini batch size = ', mini_batch_size, ', learning rate = ',lr)
    NeuralNetworks(train_set, train_size, epoch, mini_batch_size, lr, input_layer_size, hidden_layer_size, output_layer_size, train_set_label)
print('-------------------------------------\n')

print('NEURAL NETWORKS WITH DIFFERENT MINI-BATCH SIZE\n')
for size in mini_batch_size_arr :
    print('epoch = ',epoch, ', mini batch size = ', size, ', learning rate = ', learning_rate)
    NeuralNetworks(train_set, train_size, epoch, size, learning_rate, input_layer_size, hidden_layer_size, output_layer_size, train_set_label)
print('-------------------------------------\n')

print('NEURAL NETWORKS WITH DIFFERENT NUMBER OF EPOCHS\n')
for e in epoch_arr :
    print('epoch = ',e, ', mini batch size = ', mini_batch_size, ', learning rate = ', learning_rate)
    NeuralNetworks(train_set, train_size, e, mini_batch_size, learning_rate, input_layer_size, hidden_layer_size, output_layer_size, train_set_label)
print('-------------------------------------\n')