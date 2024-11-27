import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
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
		self.resnet_output_size = self.resnet.config.hidden_sizes[-1]
		self.hidden_size = hidden_size
		self.embedding_dim = embedding_dim
		self.num_layers = num_layers

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
		self.feature_proj = nn.Linear(self.resnet_output_size, embedding_dim) # For image features
		self.output_layer = nn.Linear(hidden_size, vocab_size) # For LSTM outputs
	
	def forward(self, images, captions=None):
		with torch.no_grad():
			features = self.resnet(images).last_hidden_state # (bs, channels, h, w)
		batch_size = features.size(0)
		device = images.device

		features = features.mean(dim=[2, 3]).unsqueeze(1)  # (bs, 1, channels)
		image_embeddings = self.feature_proj(features)

		hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
				torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

		if captions is not None:
			# Training phase with teacher forcing
			embeddings = self.embedding(captions)  # (bs, seq_len, embedding_dim)

			embeddings = torch.cat((image_embeddings, embeddings), dim=1)  # (bs, seq_len+1, embedding_dim)

			lstm_out, hidden = self.lstm(embeddings, hidden)  # (bs, seq_len+1, hidden_size)
			outputs = self.output_layer(lstm_out)  # (bs, seq_len+1, vocab_size)
			# Exclude first output (corresponding to the image embedding)
			outputs = outputs[:, 1:, :] # (bs, seq_len, vocab_size)
			return outputs
		else:
			# Inference phase
			return self.generate_caption(images, hidden, image_embeddings)

	def generate_caption(self, images, hidden, image_embeddings, max_seq_len=20):
		batch_size = images.size(0)
		device = images.device

		inputs = torch.zeros(batch_size, 1).long().to(device)  # Assuming <start> token index is 0

		# Initial input is the image embedding
		embeddings = image_embeddings  # (bs, 1, embedding_dim)

		captions = []

		for _ in range(max_seq_len):
			lstm_out, hidden = self.lstm(embeddings, hidden)  # (bs, 1, hidden_size)
			outputs = self.linear(lstm_out.squeeze(1))  # (bs, vocab_size)
			_, predicted = outputs.max(1)  # (bs,)
			captions.append(predicted.unsqueeze(1))  # (bs, 1)
			inputs = predicted.unsqueeze(1)
			embeddings = self.embedding(inputs)  # (bs, 1, embedding_dim)

		captions = torch.cat(captions, 1)  # (bs, max_seq_len)
		return captions  # Return the predicted captions


def train_resnetLSTMCaption(
	model: ResnetLSTMCaption,
	dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler._LRScheduler,
	criterion: nn.Module,
	device: torch.device,
	num_epochs: int,
	log_wandb: bool = True,
):
	model.train()
	for epoch in range(num_epochs):
		epoch_loss = 0
		for images, captions in dataloader:
			images, captions = images.to(device), captions.to(device)

			# Shift captions by one for input and target
			inputs = captions[:, :-1]  # All tokens except the last
			targets = captions[:, 1:]  # All tokens except the first

			# Forward pass
			outputs = model(images, captions=inputs)  # Outputs: [batch_size, caption_length, vocab_size]

			# Compute loss
			outputs = outputs.reshape(-1, outputs.size(-1))  # [batch_size * caption_length, vocab_size]
			targets = targets.reshape(-1)  # [batch_size * caption_length]
			loss = criterion(outputs, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			if log_wandb:
				wandb.log({"batch_loss": loss.item()})

		if scheduler is not None:
			scheduler.step()
		avg_epoch_loss = epoch_loss / len(dataloader)
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
		if log_wandb:
			wandb.log({"epoch_loss": avg_epoch_loss})
