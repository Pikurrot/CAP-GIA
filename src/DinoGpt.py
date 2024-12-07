import wandb
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor, GPT2LMHeadModel, GPT2Tokenizer
import evaluate


def contrastive_criterion(image_embeds, text_embeds, temperature=0.1):
	# Normalize embeddings to unit vectors
	image_embeds = F.normalize(image_embeds, p=2, dim=1)
	text_embeds = text_embeds.mean(dim=1)
	text_embeds = F.normalize(text_embeds, p=2, dim=1)

	# Compute similarity matrix (batch_size x batch_size)
	logits = torch.mm(image_embeds, text_embeds.t()) / temperature

	# Targets are diagonal (positive pairs)
	targets = torch.arange(logits.size(0), device=logits.device)

	# Compute cross-entropy loss
	loss = F.cross_entropy(logits, targets)
	return loss


class DinoGpt(nn.Module):
	def __init__(
			self,
			output_dir: str
	):
		super().__init__()
		
		# Load pre-trained DINO model
		self.encoder_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", cache_dir=output_dir)
		self.encoder = AutoModel.from_pretrained("facebook/dinov2-small", cache_dir=output_dir)
		# Allow fine-tuning of the DINO encoder (unfreeze all layers)
		for param in self.encoder.parameters():
			param.requires_grad = True

		# Load GPT-2 model
		self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=output_dir)
		self.decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=output_dir)
		if self.decoder_tokenizer.pad_token is None:
			self.decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
			self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

		# Project image embedding to GPT-2 embedding dimension
		self.proj = nn.Sequential(
			nn.Linear(self.encoder.config.hidden_size, self.decoder.config.n_embd),
			nn.ReLU(),
			nn.Linear(self.decoder.config.n_embd, self.decoder.config.n_embd)
		)

	def forward(
			self,
			images: list, # PIL images
			captions: list[str] = None,
			max_length: int=30,
			repetition_penalty: float=1.2,
			alpha: float=2.0, # Image embedding weight
			lambda_contrastive: float=0.1
	):
		device = next(self.parameters()).device

		# Encode images with DINO
		pixel_values = self.encoder_processor(images, return_tensors="pt").pixel_values.to(device)
		dino_outputs = self.encoder(pixel_values=pixel_values)
		
		# Average pool the output of the last layer
		image_embeds = dino_outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]

		# Project to GPT embedding size
		image_embeds = self.proj(image_embeds) # [B, n_embd]
		image_embeds = alpha * image_embeds

		if captions is not None:
			# Training/Validation Mode
			# Tokenize captions
			encodings = self.decoder_tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
			input_ids = encodings.input_ids.to(device)    # [B, L]
			attention_mask = encodings.attention_mask.to(device)

			# Create inputs_embeds by concatenating image_embeds as a prefix
			text_embeds = self.decoder.transformer.wte(input_ids)  # [B, L, n_embd]
			inputs_embeds = torch.cat([image_embeds.unsqueeze(1), text_embeds], dim=1)  # [B, 1+L, n_embd]

			# Adjust labels to match outputs
			labels = input_ids.clone()
			labels = torch.cat([torch.full((labels.size(0), 1), -100, device=device), labels], dim=1)

			# Pass through GPT2
			outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1), labels=labels)
			
			# Compute loss
			image_latent = F.normalize(image_embeds, p=2, dim=1) # Normalize to unit length
			text_latent = F.normalize(text_embeds, p=2, dim=1)
			cross_entropy_loss = outputs.loss
			contrastive_loss = contrastive_criterion(image_latent, text_latent)
			total_loss = cross_entropy_loss + lambda_contrastive * contrastive_loss
			return total_loss

		else:
			# Inference Mode
			B = image_embeds.shape[0]

			# Start each sequence with BOS token
			bos_token_id = self.decoder_tokenizer.bos_token_id if self.decoder_tokenizer.bos_token_id is not None else self.decoder_tokenizer.eos_token_id
			bos_tokens = torch.full((B, 1), bos_token_id, device=device)  # [B, 1]
			generated = bos_tokens
			done = torch.zeros(B, dtype=torch.bool, device=device)

			past_tokens = None  # Keep track of generated tokens for applying repetition penalty

			for _ in range(max_length):
				# Embed current tokens
				text_embeds = self.decoder.transformer.wte(generated)  # [B, seq_len, n_embd]
				
				# Concatenate image embeddings at the front
				combined_embeds = torch.cat([image_embeds.unsqueeze(1), text_embeds], dim=1)  # [B, 1+seq_len, n_embd]

				# GPT forward
				outputs = self.decoder(inputs_embeds=combined_embeds)
				logits = outputs.logits[:, -1, :]  # [B, vocab_size]

				# Apply repetition penalty to logits
				if past_tokens is not None:
					for batch_idx in range(B):
						for token in past_tokens[batch_idx]:
							logits[batch_idx, token] /= repetition_penalty

				# Greedy next token
				next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]

				# Append next token to sequences
				generated = torch.cat([generated, next_token], dim=1)  # [B, seq_len+1]

				# Track tokens for repetition penalty
				if past_tokens is None:
					past_tokens = next_token
				else:
					past_tokens = torch.cat([past_tokens, next_token], dim=1)

				# Check for EOS
				eos_mask = (next_token.squeeze(-1) == self.decoder_tokenizer.eos_token_id)
				done = done | eos_mask
				if done.all():
					break

			# Decode the generated tokens for each sequence
			generated_texts = []
			for seq in generated:
				if seq[0].item() == bos_token_id:
					seq = seq[1:]
				text = self.decoder_tokenizer.decode(seq, skip_special_tokens=True)
				generated_texts.append(text.strip())

			return generated_texts


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
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
