import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor, GPT2LMHeadModel, GPT2Tokenizer
import evaluate

class DinoGpt(nn.Module):
	def __init__(
			self,
			output_dir: str
	):
		super(DinoGpt, self).__init__()
		self.output_dir = output_dir

		# Load pre-trained DINO model
		self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", cache_dir=self.output_dir)
		self.dino = AutoModel.from_pretrained("facebook/dinov2-small", cache_dir=self.output_dir)

		# Load pre-trained GPT-2 model
		self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=self.output_dir)
		self.gpt_tokenizer.bos_token = "<start>"
		self.gpt_tokenizer.eos_token = "<end>"
		self.gpt_tokenizer.pad_token = "[PAD]"
		self.gpt = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=self.output_dir)
		self.gpt.resize_token_embeddings(len(self.gpt_tokenizer))

		# Freeze DINO parameters
		# for param in self.dino.parameters():
		# 	param.requires_grad = False
		# for layer in self.dino.encoder.layer[-2:]:
		# 	for param in layer.parameters():
		# 		param.requires_grad = True

		# Linear projection layer
		self.proj = nn.Sequential(
			nn.Linear(self.dino.config.hidden_size, self.gpt.config.n_embd),
			nn.ReLU(),
			nn.Linear(self.gpt.config.n_embd, self.gpt.config.n_embd)
		)

	def forward(
			self,
			images: torch.Tensor, # (bs, channels, h, w)
			captions: torch.Tensor = None, # (bs, seq_len)
			max_seq_len: int = 20
	):
		# Extract image features
		with torch.no_grad():
			inputs = self.dino_processor(images, return_tensors="pt", padding=True, truncation=True)
			inputs = inputs.to(self.dino.device)
			image_features = self.dino(**inputs)[0][:, 0, :] # (bs, hidden_size)

		# Project image features to GPT-2 embedding size
		image_embeddings = self.proj(image_features).unsqueeze(1) # (bs, 1, n_embd)

		if captions is not None:
			# Training with teacher forcing

			# Preprocess captions
			captions = [f"<start> {caption.lower().strip()} <end>" for caption in captions]

			# Tokenize captions
			encoding = self.gpt_tokenizer(
				captions,
				return_tensors="pt",
				padding=True,
				truncation=True,
				add_special_tokens=True
			)
			input_ids = encoding.input_ids.to(self.dino.device)  # (bs, seq_len)
			attention_mask = encoding.attention_mask.to(self.dino.device)  # (bs, seq_len)

			# Get input embeddings
			inputs_embeds = self.gpt.transformer.wte(input_ids)  # (bs, seq_len, n_embd)

			# Concatenate image embeddings with inputs
			inputs_embeds = torch.cat((image_embeddings, inputs_embeds[:, :-1, :]), dim=1)  # Shape: (bs, seq_len + 1, n_embd)

			# Update attention mask
			attention_mask = torch.cat(
				(torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype),
				attention_mask[:, :-1]),
				dim=1
			)  # (bs, seq_len + 1)

			# Labels
			labels = input_ids

			# Compute loss
			outputs = self.gpt(
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				labels=labels
			)
			loss = outputs.loss
			return loss
		else:
			# Inference
			batch_size = image_embeddings.size(0)

			# Get <start> token embedding
			start_token_id = self.gpt_tokenizer.bos_token_id
			start_token_embedding = self.gpt.transformer.wte(
				torch.tensor([start_token_id], device=image_embeddings.device)
			).unsqueeze(0)  # (1, 1, n_embd)
			start_token_embeddings = start_token_embedding.repeat(batch_size, 1, 1)  # (bs, 1, n_embd)

			# Concatenate image embeddings and start token embedding
			inputs_embeds = torch.cat((image_embeddings, start_token_embeddings), dim=1)  # (bs, 2, n_embd)

			# Update attention mask
			attention_mask = torch.ones((batch_size, 2), dtype=torch.long, device=image_embeddings.device)  # (bs, 2)

			# Generate captions
			generated_ids = self.gpt.generate(
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				max_new_tokens=max_seq_len,
				num_beams=5,
    			early_stopping=True,
				pad_token_id=self.gpt_tokenizer.pad_token_id,
				eos_token_id=self.gpt_tokenizer.eos_token_id
			)

			# Remove image and <start> token ids
			generated_ids = generated_ids[:, 2:]  # (bs, generated_seq_len)

			# Decode the generated tokens
			captions = self.gpt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

			return captions


def train_DinoGpt(
		model: DinoGpt,
		train_loader: DataLoader,
		val_loader: DataLoader,
		optimizer: torch.optim.Optimizer,
		scheduler: torch.optim.lr_scheduler._LRScheduler,
		device: torch.device,
		num_epochs: int,
		log_wandb: bool = True
):
	# Evaluation Metrics
	bleu = evaluate.load("bleu")
	meteor = evaluate.load("meteor")
	rouge = evaluate.load("rouge")

	for epoch in range(num_epochs):
		# Training Phase
		model.train()
		train_loss = 0
		for images, captions in train_loader:		
			# Forward pass
			loss = model(images, captions)
			train_loss += loss.item()
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			optimizer.step()
			
			if log_wandb:
				wandb.log({"train_batch_loss": loss.item()})
			print(f"Batch Loss: {loss.item():.4f}")
		
		# Validation Phase
		model.eval()
		val_loss = 0
		predictions, references = [], []
		with torch.no_grad():
			for images, captions in val_loader:			
				# Compute loss
				loss = model(images, captions)
				val_loss += loss.item()
				
				# Generate captions
				pred_captions = model(images, captions=None)
				predictions.extend(pred_captions)
				references.extend(captions)
		
		# Scheduler Step
		if scheduler:
			scheduler.step()

		# Print some random examples
		print()
		for i in np.random.randint(0, len(predictions), 5):
			print(f"Prediction: {predictions[i]}")
			print(f"Reference: {references[i]}")
			print()
		
		# Compute Metrics
		res_bleu_1 = bleu.compute(predictions=predictions, references=[[ref] for ref in references], max_order=1)
		res_bleu_2 = bleu.compute(predictions=predictions, references=[[ref] for ref in references], max_order=2)
		res_meteor = meteor.compute(predictions=predictions, references=[[ref] for ref in references])
		res_rouge = rouge.compute(predictions=predictions, references=[[ref] for ref in references])
		
		# Log Metrics
		avg_train_loss = train_loss / len(train_loader)
		avg_val_loss = val_loss / len(val_loader)
		
		print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
		if log_wandb:
			wandb.log({
				"epoch_train_loss": avg_train_loss,
				"epoch_val_loss": avg_val_loss,
				"BLEU-1": res_bleu_1["bleu"],
				"BLEU-2": res_bleu_2["bleu"],
				"ROUGE-L": res_rouge["rougeL"],
				"METEOR": res_meteor["meteor"],
			})
