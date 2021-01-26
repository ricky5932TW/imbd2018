#coding:utf-8
import os  
import xlrd


import torch
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math

test_path = "test_10/"
train_path = "train_40/"

data_Spindle_X = []
data_Spindle_Y = []
data_Workbench_X = []
data_Workbench_Y = []
data_label = []

test_Spindle_X = []
test_Spindle_Y = []
test_Workbench_X = []
test_Workbench_Y = []
test_label = []

def getData(file_path):

        workbook = xlrd.open_workbook(file_path)
        sheet = workbook.sheets()[0]

        sheet_data = {
                'Spindle_X':[],
                'Spindle_Y':[],
                'Workbench_X':[], 
                'Workbench_Y':[]
        }
        
        for i in range(7500):
                sheet_data['Spindle_X'].append(sheet.cell(i,0).value)
                sheet_data['Spindle_Y'].append(sheet.cell(i,1).value)
                sheet_data['Workbench_X'].append(sheet.cell(i,2).value)
                sheet_data['Workbench_Y'].append(sheet.cell(i,3).value)
        
    
        if 'test' in file_path:
                test_Spindle_X.append([sheet_data['Spindle_X']])
                test_Spindle_Y.append([sheet_data['Spindle_Y']])
                test_Workbench_X.append([sheet_data['Workbench_X']])
                test_Workbench_Y.append([sheet_data['Workbench_Y']])
                test_label.append([float(sheet.cell(7500,0).value[3:])])
        else:
		data_Spindle_X.append([sheet_data['Spindle_X']])
                data_Spindle_Y.append([sheet_data['Spindle_Y']])
                data_Workbench_X.append([sheet_data['Workbench_X']])
                data_Workbench_Y.append([sheet_data['Workbench_Y']])
                data_label.append([float(sheet.cell(7500,0).value[3:])])

for file_name in os.listdir(test_path):
        print file_name
        getData(test_path+file_name)

for file_name in os.listdir(train_path):
        print file_name
        getData(train_path+file_name)

#RNN2


trainX = torch.tensor(data_Spindle_Y)
trainY = torch.tensor(np.reshape(np.array(data_label),(40,1))).float()

print(trainX.size())
print(trainY.size())


testX = torch.tensor(test_Spindle_Y)
testY = torch.tensor(np.reshape(np.array(test_label),(10,1))).float()


print(testX.size())
print(testY.size())

# NN3------------------------------------------------------------

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(1, 2, kernel_size=5,padding=1)
        self.conv2 = torch.nn.Conv1d(2, 4, kernel_size=5,padding=1)
        self.mp = torch.nn.MaxPool1d(2)
        self.hidden = torch.nn.Linear(7492, n_hidden)   # hidden layer
        self.hidden_1 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):

        in_size = x.clone().size(0)

	x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = x.view(in_size,-1)
        
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=len(trainX[0]), n_hidden=256, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.045)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

all_error = []


b_x = trainX
b_y = trainY
test_x = testX
test_y = testY

all_rmse = []
for epoch in range(100):
    print(epoch)
    all_pred = []
    all_target = []
        
    output = net(b_x)                               # rnn output
    loss = loss_func(output, b_y)                   # cross entropy loss
    optimizer.zero_grad()                           # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()
    #break
    output = net(test_x)
    count = 0.0
    sum = 0
    for i in range(len(output)):
	if(test_y[i]!=0):
        	num = abs(float(output[i]) - float(test_y[i])) / float(test_y[i])
        else:
		num = 0
	sum = sum + pow((float(output[i]) - float(test_y[i])),2)
        all_pred.append(round(float(output[i]),6))
        all_target.append(round(float(test_y[i]),6))
        if num <= 0.1:
            count = count + 1.0
    sum = sum / len(output)
    rmse = math.sqrt(sum)
    all_rmse.append(rmse)
    print('Target:')
    print(all_target)
    print('Pred:')
    print(all_pred)
    print('RMSE:')
    print(float(rmse))
    print('Count:')
    print(round((count / len(output)) * 100,2))
    print('======================================')

    #break
print('END')



