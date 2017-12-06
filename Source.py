import matplotlib.pyplot as plt
import scipy.special
import numpy
import csv

#%matplotlib inline - This is available only for the Jupyter Notebook.

#======Class-Space==============================================================
class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)    #Sigmoid
#        self.activation_function = lambda x: x * (1.0 - x)             #dSigmoid
#        self.activation_function = lambda x: numpy.tanh(x)             #hyperbolic tangent
#        self.activation_function = lambda x: x * (x > 0)               #ReLU

        self.lr = learningRate
        pass

    '''
    def activation_function(self, x):    #SoftMax
        e = numpy.exp(x - numpy.max(x)) #prevent overload
        if e.ndim == 1:
            return e / numpy.sum(e, axis=0)
        else:
            return e / numpy.array([numpy.sum(e, axis=1)]).T    #ndim = 2
        pass
    '''

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outpus = self.activation_function(final_inputs)

        #calculate difference between expectation and result
        output_errors = targets - final_outpus
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #Actualization of weights for synapses between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                            (1.0 - hidden_outputs)), numpy.transpose(inputs))

        #Actualization of weights for synapses between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outpus *
                        (1.0 - final_outpus)), numpy.transpose(hidden_outputs))
        pass

    #Test out neural network, testing images data set
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outpus = self.activation_function(final_inputs)

        return final_outpus
    pass
#======Class-Space=END==========================================================


#---Initialization-of-class-object----------------------------------------------
input_nodes = 784       #input_nodes equel: 28 x 28 pixels
hidden_nodes = 10       #hidden_nodes equel: 10 number [0 - 9]
output_nodes = 10       #output_nodes equel: 10 number [0 - 9]

learning_rate = 0.1

N = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#---Initialization-of-class-object-END------------------------------------------

#---Training-neural-network-by-mnist_train-file---------------------------------
training_data_file = open('Mnist/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 1
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        N.train(inputs, targets)
        pass
    pass
#---Training-neural-network-by-mnist_train-file-END-----------------------------

#---Testing-neural-network-by-mnist_test-file-----------------------------------
test_data_file = open('Mnist/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',')
print('[', training_data_list[0], '] Training data')

image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')

N.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

scoreCard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])

    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = N.query(inputs)
    label = numpy.argmax(outputs)
    print('Correct[{c}] ===> [{l}]Selected'.format(c = correct_label, l = label))

    if (label == correct_label):
        scoreCard.append(1)
    else:
        scoreCard.append(0)
        pass
    pass
#---Testing-neural-network-by-mnist_test-file-END-------------------------------

#===Convert-Data-And-Put-Into-File==============================================
scoreCardArray = numpy.asarray(scoreCard)
performance = scoreCardArray.sum() / scoreCardArray.size
print("Perfomance equel: [", (performance * 100), '] percents!')

performance_csv = []
performance_csv.append(performance)
performance_csv.append(learning_rate)
performance_csv.append(hidden_nodes)
performance_csv.append(epochs)
performance_csv.append(';  ')
print('Perfomance, learnRate, hNodes, epochs')
print(performance_csv)

csv_file = open('performance_history.csv', 'a', newline='')
wr = csv.writer(csv_file)
wr.writerows([performance_csv])
csv_file.close()
#===Convert-Data-And-Put-Into-File=END==========================================
