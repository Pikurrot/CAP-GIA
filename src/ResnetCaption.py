import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

class ResnetLSTMCaption(nn.Module):
	def __init__(
			self,
			resnet_name: str,
			output_dir: str,
			hidden_size: int,
			vocab_size: int,
			embedding_dim: int,
			num_layers: int
		):
		super(ResnetLSTMCaption, self).__init__()
		self.model_name = "microsoft/" + resnet_name
		self.output_dir = output_dir

		# Load pre-trained ResNet model
		self.resnet = AutoModel.from_pretrained(self.model_name, cache_dir=self.output_dir)
		
		# Freeze ResNet parameters
		for param in self.resnet.parameters():
			param.requires_grad = False

		# Define LSTM
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(
			embedding_dim,
			hidden_size,
			num_layers,
			batch_first=True
		)
		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, images, max_caption_length=20):
		# Extract features from images using ResNet
		with torch.no_grad():
			features = self.resnet(images).last_hidden_state

		# Initialize LSTM state
		batch_size = features.size(0)
		hidden = (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(features.device),
					torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(features.device))

		# Start with the <start> token (assuming it's index 0)
		inputs = torch.zeros(batch_size, 1).long().to(features.device)

		# Generate captions
		captions = []
		for _ in range(max_caption_length):
			embeddings = self.embedding(inputs)
			lstm_out, hidden = self.lstm(embeddings, hidden)
			outputs = self.linear(lstm_out.squeeze(1))
			_, predicted = outputs.max(1)
			captions.append(predicted.unsqueeze(1))
			inputs = predicted.unsqueeze(1)

		captions = torch.cat(captions, 1)
		return captions
