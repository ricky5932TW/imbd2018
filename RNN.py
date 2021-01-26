import os
import csv
import xlrd
import random
from random import randint
from pymongo import MongoClient

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
	
#RNN1 
	
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
	
	for i in range(5):
		second_data = {
			'Spindle_X':[],
			'Spindle_Y':[],
			'Workbench_X':[],
			'Workbench_Y':[]
		}

		for j in range(1500):
			second_data['Spindle_X'].append(sheet.cell(i*1500+j,0).value)
			second_data['Spindle_Y'].append(sheet.cell(i*1500+j,1).value)
			second_data['Workbench_X'].append(sheet.cell(i*1500+j,2).value)
			second_data['Workbench_Y'].append(sheet.cell(i*1500+j,3).value)
		sheet_data['Spindle_X'].append(second_data['Spindle_X'])
		sheet_data['Spindle_Y'].append(second_data['Spindle_Y'])
		sheet_data['Workbench_X'].append(second_data['Workbench_X'])
		sheet_data['Workbench_Y'].append(second_data['Workbench_Y'])		
	if 'test' in file_path:
		test_Spindle_X.append(sheet_data['Spindle_X'])
		test_Spindle_Y.append(sheet_data['Spindle_Y'])
		test_Workbench_X.append(sheet_data['Workbench_X'])
		test_Workbench_Y.append(sheet_data['Workbench_Y'])
		test_label.append(float(sheet.cell(7500,0).value[3:]))
	else:
		data_Spindle_X.append(sheet_data['Spindle_X'])
		data_Spindle_Y.append(sheet_data['Spindle_Y'])
		data_Workbench_X.append(sheet_data['Workbench_X'])
		data_Workbench_Y.append(sheet_data['Workbench_Y'])
		data_label.append(float(sheet.cell(7500,0).value[3:]))
		
for file_name in os.listdir(test_path):
	print file_name
	getData(test_path+file_name)
	
for file_name in os.listdir(train_path):
	print file_name
	getData(train_path+file_name)

	
#RNN2

trainX = []
trainY = []
count = 0
index = 10
for i in range(4):
    trainX.append(data_Spindle_Y[count:(count+index)])
    trainY.append(data_label[count:(count+index)])
    count += 10
    
trainX = torch.tensor(trainX)
trainY = torch.tensor(trainY)

print(trainX.size())
print(trainY.size())

testX = []
testY = []
count = 0
index = 10
for i in range(1):
    testX.append(test_Spindle_Y[count:(count+index)])
    testY.append(test_label[count:(count+index)])
    
testX = torch.tensor(testX)
testY = torch.tensor(testY)

print(testX.size())
print(testY.size())
	
#RNN3

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 100               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 10
TIME_STEP = 5          # rnn time step / image height
INPUT_SIZE = 1500         # rnn input size / image width
LR = 0.01               # learning rate

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=128,         # rnn hidden unit
            num_layers=4,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        
	r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()                       # the target label is not one-hotted

# training and testing
test_x = testX.view(-1,TIME_STEP,INPUT_SIZE)
test_y = testY[0].view(-1,1)

all_rmse = []
for epoch in range(EPOCH):
    print(epoch)
    all_pred = []
    all_target = []
    for step, x in enumerate(trainX): 
        b_x = x.view(BATCH_SIZE,TIME_STEP,INPUT_SIZE)
        b_y = trainY[step].view(BATCH_SIZE,1)
        
        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()
        #break
    output = rnn(test_x)
    count = 0.0
    sum = 0
    for i in range(len(output)):
        num = abs(float(output[i]) - float(test_y[i])) / float(test_y[i])
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


