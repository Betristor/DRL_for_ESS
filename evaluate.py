import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from pickle import load

class LSTMCNNModel(nn.Module):
	def __init__(self):
		super(LSTMCNNModel ,self).__init__()
		self.cnn = nn.Conv1d(8, 24, kernel_size=24, stride=24)
		self.lstm = nn.LSTM(7, 32, 6, batch_first=True)
		self.fc = nn.Linear(32, 1)
		self.relu = nn.ReLU()
		# self.device = device

	def forward(self, x):
		out = self.relu(self.cnn(x))
		out, states = self.lstm(out)
		out = self.fc(out)
		return out

# Model class must be defined somewhere
model = LSTMCNNModel()
model.load_state_dict(torch.load("/Users/yuhengzhang/Documents/博一上/Foundations of RL/Project/DRL_for_ESS/models/da_price_prediction.pt"))
model.eval()

train_set_load = open("/Users/yuhengzhang/Documents/博一上/Foundations of RL/Project/DRL_for_ESS/data/processed_data/train_data.pkl", "rb") 
train_set = load(train_set_load)
train_set_load.close()

inputs = np.moveaxis(train_set['X_train'], -1, 1)
inputs = torch.tensor(inputs, dtype=torch.float64)
print(inputs.shape)
with torch.no_grad():
    pass