import os  
import xlrd

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
		test_Spindle_X.append(sheet_data['Spindle_X'])
		test_Spindle_Y.append(sheet_data['Spindle_Y'])
		test_Workbench_X.append(sheet_data['Workbench_X'])
		test_Workbench_Y.append(sheet_data['Workbench_Y'])
		test_label.append([float(sheet.cell(7500,0).value[3:])])
	else:
                data_Spindle_X.append(sheet_data['Spindle_X'])
                data_Spindle_Y.append(sheet_data['Spindle_Y'])
                data_Workbench_X.append(sheet_data['Workbench_X'])
                data_Workbench_Y.append(sheet_data['Workbench_Y'])
                data_label.append([float(sheet.cell(7500,0).value[3:])])

for file_name in os.listdir(test_path):
        print file_name
        getData(test_path+file_name)

for file_name in os.listdir(train_path):
        print file_name
        getData(train_path+file_name)

#RNN2

trainX = torch.tensor(data_Spindle_X)
trainY = torch.tensor(data_label)

print data_label[0]
print(trainX.size())
print(trainY.size())

testX = torch.tensor(test_Spindle_X)
testY = torch.tensor(test_label)

print(testX.size())
print(testY.size())


# NN3------------------------------------------------------------

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden_1 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=len(trainX[0]), n_hidden=256, n_output=1)     # define the network
print 'n_feature=',len(trainX[0])

optimizer = torch.optim.SGD(net.parameters(), lr=0.045)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

all_error = []

for t in range(1,1001):
    prediction = net(trainX)     # input x and predict based on x
    #print 'prediction',prediction,'trainY',trainY
    loss = loss_func(prediction, trainY)     # must be (1. nn output, 2. target)
    #print loss
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 100 == 0:
        
        k = prediction.data.numpy()
        ans = trainY.numpy()
        real = []
        right = []
        error = []
        count = 0.0
        for i in range(len(ans)):
            num = abs(ans[i][0] - k[i][0]) / ans[i]
            real.append(ans[i][0])
            right.append(k[i][0])
            error.append(float(num))
            if num < 0.1:
                count = count + 1.0
        print('Epoch: ',t,'| Train Loss:',float(loss.data.numpy()),'| Acc:',(count / len(ans)) * 100,'%')
        all_error.append(error)
        print '=========================================================='

print 'END'
torch.save(net,'CNC_tcim_20180715.pkl')


# NN4------------------------------------------------------------

net = torch.load('CNC_tcim_20180715.pkl')
prediction = net(testX)     # input x and predict based on x

loss = loss_func(prediction, testY)     # must be (1. nn output, 2. target)
print loss.data.numpy()
k = prediction.data.numpy()
ans = testY.numpy()
real = []
right = []
error = []
count = 0.0
for i in range(len(ans)):
    num = abs(ans[i][0] - k[i][0]) / ans[i]
    real.append(ans[i][0])
    right.append(round(k[i][0],8))
    error.append(round(float(num),4))
    if num < 0.1:
        count = count + 1.0
print 'ans',real
print 'test',right
print 'error',error
print (count / len(ans)) * 100,'%'
print '==========================='



