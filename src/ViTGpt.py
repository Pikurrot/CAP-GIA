import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel, GPT2LMHeadModel, GPT2Tokenizer
import evaluate

class ViTGpt(nn.Module):
	def __init__(
			self,
			output_dir: str
	):
		super().__init__()

		# Load Vision Transformer
		self.encoder_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", cache_dir=output_dir)
		self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224", cache_dir=output_dir)
		self.encoder.eval()  # Usually keep it frozen
		for param in self.encoder.parameters():
			param.requires_grad = False

		# Load GPT-2
		self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=output_dir)
		self.decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=output_dir)
		if self.decoder_tokenizer.pad_token is None:
			self.decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
			self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

		# Project image embedding to GPT-2 embedding dimension
		self.proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.n_embd)

	def forward(self, images, captions=None, max_length=30):
		device = next(self.parameters()).device
		
		# 1. Encode images with ViT
		with torch.no_grad():
			pixel_values = self.encoder_processor(images, return_tensors="pt").pixel_values.to(device)
			vit_outputs = self.encoder(pixel_values=pixel_values)
			# Take the CLS token embedding
			image_embeds = vit_outputs.last_hidden_state[:,0,:]  # [B, hidden_size]
		# Project to GPT embedding size
		image_embeds = self.proj(image_embeds)  # [B, n_embd]

		if captions is not None:
			# Training/Validation Mode
			# Tokenize captions
			encodings = self.decoder_tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
			input_ids = encodings.input_ids.to(device)    # [B, L]
			attention_mask = encodings.attention_mask.to(device)

			# Create input_embeds by concatenating image_embeds as a prefix
			# image_embeds: [B, n_embd]
			# We need to expand image_embeds to [B, 1, n_embd] and then concat with GPT embeddings
			input_embeds = self.decoder.transformer.wte(input_ids)  # [B, L, n_embd]
			image_embeds = image_embeds.unsqueeze(1)            # [B, 1, n_embd]
			inputs_embeds = torch.cat([image_embeds, input_embeds], dim=1)  # [B, 1+L, n_embd]

			# Adjust labels to match outputs
			# The labels should be shifted by one inside GPT to compute the loss.
			# If we feed [image_embed + caption_tokens], we want GPT to predict caption_tokens shifted by one.
			# Let's shift the label by 1 position.
			labels = input_ids.clone()
			# For GPT-2, when we prepend image_embeds, it means first token is not a real text token.
			# We should shift labels accordingly by inserting a -100 at the front (ignore index).
			# The first output corresponds to image_embeds, there's no label for that.
			# So we insert -100 at the start of each sequence of labels:
			labels = torch.cat([torch.full((labels.size(0), 1), -100, device=device), labels], dim=1)

			# Pass through GPT2
			outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1), labels=labels)
			loss = outputs.loss
			return loss

		else:
			# Inference Mode
			# We start from image_embeds and a BOS token or a known token
			# GPT-2 doesn't have a specific BOS token, often <|endoftext|> is used as start.
			bos_token_id = self.decoder_tokenizer.bos_token_id if self.decoder_tokenizer.bos_token_id is not None else self.decoder_tokenizer.eos_token_id
			if bos_token_id is None:
				# If GPT-2 tokenizer doesn't have bos_token_id, use the eos_token_id for start
				bos_token_id = self.decoder_tokenizer.eos_token_id

			# Start the generation loop
			generated = torch.tensor([[bos_token_id]], device=device)  # [B, 1]
			for _ in range(max_length):
				# Embed current tokens
				input_embeds = self.decoder.transformer.wte(generated) # [B, seq_len, n_embd]

				# Combine with image_embeds prefix
				combined_embeds = torch.cat([image_embeds.unsqueeze(1), input_embeds], dim=1) # [B, 1+seq_len, n_embd]

				# GPT forward
				outputs = self.decoder(inputs_embeds=combined_embeds)
				# Get logits for the last token
				logits = outputs.logits[:, -1, :] # [B, vocab_size]

				# Greedy sampling
				next_token = torch.argmax(logits, dim=-1).unsqueeze(-1) # [B, 1]

				# Append next token to generated
				generated = torch.cat([generated, next_token], dim=1)

				# Stop if next_token == eos_token_id
				if next_token.item() == self.decoder_tokenizer.eos_token_id:
					break
			
			# Decode the generated tokens (excluding the bos token)
			generated_texts = []
			for seq in generated:
				# Remove the bos token and decode
				seq = seq[1:] if seq[0] == bos_token_id else seq
				text = self.decoder_tokenizer.decode(seq, skip_special_tokens=True)
				generated_texts.append(text.strip())
			
			return generated_texts

def train_ViTGpt(
		model: ViTGpt,
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
