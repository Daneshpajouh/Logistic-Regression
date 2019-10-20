%reset
import numpy as np
import random
import matplotlib.pyplot as plt
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
data = np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Logistic-Regression/master/iris.data')), dtype='str', delimiter = ',')
classes = le.fit(data[:,-1]).classes_
x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1:], test_size=0.33, random_state=42)
m = len(y_train)
x_train = np.c_[np.ones(m), (np.array(x_train, dtype= float))]
x_test = np.c_[np.ones(len(y_test)), (np.array(x_test, dtype= float))]

# Initializing required values
theta = np.zeros((len(x_train.transpose()),1))
alpha = 0.17
m = len(y_train)
lam = 0.009
num_itr = 200
regular = True
epsilon = 1e-3
thetas = []
cost = []
y_matrix = []

def g(x):    # Sigmoid Function
    return 1 / (1 + np.exp(-x))

for c in classes:    # One vs All Classification module
    y_t = np.where(y_train==c, 1, 0)
    print(c)
    for i in range(num_itr):   # Main loop
        if regular == True:    # Model using regularization
            h_x = g(x_train @ theta)      
            error = (float(((np.log(g(x_train @ theta))).transpose() @ y_t + ((np.log(1-g(x_train @ theta))).transpose() @ (1-y_t)))/(-m))) + (((np.sum(np.power(theta, 2)))*lam)/(2*m))
            cost.append(error)
            theta = theta - (alpha * ((((x_train.transpose())@(h_x - y_t))/m) + (theta * (lam/m))))     # Training Theta via "Gradient Descent function"
            if i < 5:
                pass
            elif abs(np.mean(cost[-5:]) - error) <= epsilon:     # Break the loop if error is too small or not changing (Convergence)
                break
        else:    # Model without using regularization
            h_x = g(x_train @ theta)      # Hypothesis (h(x)) function
            error = float(((np.log(g(x_train @ theta))).transpose() @ y_t + ((np.log(1-g(x_train @ theta))).transpose() @ (1-y_t)))/(-m))
            cost.append(error)
            theta = theta - ((alpha / m) * ((x_train.transpose())@(h_x - y_t)))     # Training Theta via "Gradient Descent function"
            if i < 5:
                pass
            elif abs(np.mean(cost[-5:]) - error) <= epsilon:     # Break the loop if error is too small or not changing (Convergence)
                break
    y_matrix.append(h_x)
    thetas.append(theta)
print(thetas)

# Testing module
hypo_matrix = []
l = 0
for t in classes:    #  Testing the hypothesis with new test values using One vs All
    hypothesis = g(x_test @ thetas[l])       
    hypo_matrix.append(hypothesis)
    l += 1
hypo_max = np.argmax(hypo_matrix, axis=0)

# Accuracy module
y_prediction = np.argmax(y_matrix, axis=0)    # Finds hypothesis which returned the highest value
train = np.reshape(le.fit_transform(y_train), (len(le.fit_transform(y_train)), 1))    # Converts classes type from string to integers
accuracy = (np.count_nonzero(np.where( train == y_prediction, 1, 0))) / len(train) * 100    # Calculating training accuracy
test = np.reshape(le.fit_transform(y_test), (len(le.fit_transform(y_test)), 1))
test_accuracy = (np.count_nonzero(np.where( test == hypo_max, 1, 0))) / len(test) * 100    # Calculating testing accuracy
print('\n\nTraining Accuracy is: %', accuracy, '\n\nTesting Accuracy is: %', test_accuracy, '\n\n')

# Figure module
plot_switch = True
if plot_switch == True:
  
    n = int(((len(x_test.transpose()) - 1)*(len(x_test.transpose()) - 2))/2)    # Calculates sum of all numbers before "n"
    if n <= 3:
        n = 4
    s = n-2
    l = 0
    x = x_test
    y = y_test
    mi = np.nanmin
    ma = np.nanmax
    for i in range(n-3):    # Figure of data scatter with different colors for each class
        s -= 1
        l += 1
        for j in range(s):
            for k in range(len(classes)):
                plt.scatter(x[np.where(y == classes[k]), (i+1)], x[np.where(y == classes[k]), (l+j+1)], label = (classes[k]))
            plt.title('Iris Dataset')
            plt.xlabel(("X"+ str(i+1)), fontsize=14)
            plt.ylabel(("X"+ str(l+j+1)), fontsize=14)
            plt.axis([(mi(x_test[:,i+1])-1), (ma(x_test[:,i+1])+1), (mi(x_test[:,l+j+1])-1), (ma(x_test[:,l+j+1])+1)])
            plt.legend(loc='upper left')
            plt.show()
    
    # Accuracy figure module
    n = int(((len(x_test.transpose()) - 1)*(len(x_test.transpose()) - 2))/2)    # Calculates sum of all numbers before "n"
    if n <= 3:
        n = 4
    s = n-2
    l = 0
    y = test 
    print('\n\n\tAccuracy Scatters:\n\n')
    
    for i in range(n-3):    # Figure of data scatter with different colors for each class
        s -= 1
        l += 1
        for j in range(s):
            plt.scatter(x[np.where(y == hypo_max), (i+1)], x[np.where(y == hypo_max), (l+j+1)], label = 'Correct')
            plt.scatter(x[np.where(y != hypo_max), (i+1)], x[np.where(y != hypo_max), (l+j+1)], label = 'Wrong')
            plt.title('Accuracy Scatter')
            plt.xlabel(("X"+ str(i+1)), fontsize=14)
            plt.ylabel(("X"+ str(l+j+1)), fontsize=14)
            plt.axis([(mi(x_test[:,i+1])-1), (ma(x_test[:,i+1])+1), (mi(x_test[:,l+j+1])-1), (ma(x_test[:,l+j+1])+1)])
            plt.legend(loc='upper left')
            plt.show()
