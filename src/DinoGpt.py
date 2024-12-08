import wandb
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor, GPT2LMHeadModel, GPT2Tokenizer
import evaluate
from typing import Literal


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


def distinct_ngrams(predictions, n=1):
	ngrams = set()
	for pred in predictions:
		tokens = pred.split()
		ngrams.update(zip(*[tokens[i:] for i in range(n)]))
	return len(ngrams) / sum(len(pred.split()) for pred in predictions)


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
			lambda_contrastive: float=0.1, # Contrastive loss weight
			inference_mode: Literal["sampling", "beam_search"] = "sampling",
			sampling_threshold: float=-1, # If -1, use multinomial
			n_beams: int=5
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
			input_ids = encodings.input_ids.to(device)  # [B, L]
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
			with torch.no_grad():
				cos_sim = F.cosine_similarity(image_latent, text_latent, dim=1).mean().item()
			return total_loss, cross_entropy_loss, contrastive_loss, cos_sim, input_ids, labels, outputs.logits

		else:
			# Inference Mode
			B = image_embeds.shape[0]
			bos_token_id = self.decoder_tokenizer.bos_token_id if self.decoder_tokenizer.bos_token_id is not None else self.decoder_tokenizer.eos_token_id
			eos_token_id = self.decoder_tokenizer.eos_token_id

			if inference_mode == "sampling":
				# Sampling-based generation
				bos_tokens = torch.full((B, 1), bos_token_id, device=device)  # [B, 1]
				generated = bos_tokens
				done = torch.zeros(B, dtype=torch.bool, device=device)
				past_tokens = None  # Track past tokens for repetition penalty

				for _ in range(max_length):
					# Embed current tokens
					text_embeds = self.decoder.transformer.wte(generated)  # [B, seq_len, n_embd]

					# Concatenate image embeddings at the front
					combined_embeds = torch.cat([image_embeds.unsqueeze(1), text_embeds], dim=1)  # [B, 1+seq_len, n_embd]

					# GPT forward
					outputs = self.decoder(inputs_embeds=combined_embeds)
					logits = outputs.logits[:, -1, :]  # [B, vocab_size]

					# Apply repetition penalty
					if past_tokens is not None:
						for batch_idx in range(B):
							for token in past_tokens[batch_idx]:
								logits[batch_idx, token] /= repetition_penalty

					# Apply softmax to get probabilities
					probs = torch.softmax(logits, dim=-1)

					# Apply sampling threshold
					if sampling_threshold > 0:
						# Filter probabilities
						sorted_probs, sorted_indices = torch.sort(probs, descending=True)
						cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
						mask = cumulative_probs <= sampling_threshold
						mask[..., 1:] = mask[..., :-1].clone()
						mask[..., 0] = True
						probs *= mask
						probs /= probs.sum(dim=-1, keepdim=True)  # Re-normalize

					# Sample next token
					next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

					# Append next token to sequences
					generated = torch.cat([generated, next_token], dim=1)  # [B, seq_len+1]

					# Track tokens for repetition penalty
					if past_tokens is None:
						past_tokens = next_token
					else:
						past_tokens = torch.cat([past_tokens, next_token], dim=1)

					# Check for EOS
					eos_mask = (next_token.squeeze(-1) == eos_token_id)
					done = done | eos_mask
					if done.all():
						break

				# Decode the generated tokens
				generated_texts = []
				for seq in generated:
					if seq[0].item() == bos_token_id:
						seq = seq[1:]
					text = self.decoder_tokenizer.decode(seq, skip_special_tokens=True)
					generated_texts.append(text.strip())
				return generated_texts

			elif inference_mode == "beam_search":
				# Beam search generation
				generated_texts = []

				for b in range(B):  # Process each image independently
					# Initialize beams for this image
					beams = [([bos_token_id], 0.0, torch.full((1, 1), bos_token_id, device=device))]  # List of (tokens_list, score, tensor)
					completed = []

					for _ in range(max_length):
						new_beams = []
						for tokens, score, tensor in beams:
							# Embed current tokens
							text_embeds = self.decoder.transformer.wte(tensor)  # [1, seq_len, n_embd]

							# Concatenate image embeddings at the front
							combined_embeds = torch.cat([image_embeds[b].unsqueeze(0).unsqueeze(1), text_embeds], dim=1)  # [1, 1+seq_len, n_embd]

							# GPT forward
							outputs = self.decoder(inputs_embeds=combined_embeds)
							logits = outputs.logits[:, -1, :]  # [1, vocab_size]

							# Apply repetition penalty
							for token in tokens:
								logits[0, token] /= repetition_penalty

							# Log probabilities
							probs = torch.log_softmax(logits, dim=-1)

							# Get top-k candidates
							top_k_probs, top_k_indices = torch.topk(probs, n_beams, dim=-1)
							for prob, index in zip(top_k_probs[0], top_k_indices[0]):
								new_tokens = tokens + [index.item()]  # Append new token to the existing list
								new_score = score + prob.item()
								new_tensor = torch.cat([tensor, index.unsqueeze(0).unsqueeze(0)], dim=1)
								if index.item() == eos_token_id:
									completed.append((new_tokens, new_score))
								else:
									new_beams.append((new_tokens, new_score, new_tensor))

						# Keep the top num_beams beams
						new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:n_beams]
						beams = new_beams

						# Stop if no active beams remain
						if not beams:
							break

					# Add remaining beams to completed
					completed.extend(beams)

					# Select the highest-scoring sequence
					best_sequence = max(completed, key=lambda x: x[1])[0]
					text = self.decoder_tokenizer.decode(best_sequence, skip_special_tokens=True)
					generated_texts.append(text.strip())

				return generated_texts


def train_DinoGpt(
		model: DinoGpt,
		train_loader: DataLoader,
		val_loader: DataLoader,
		optimizer: torch.optim.Optimizer,
		scheduler: torch.optim.lr_scheduler._LRScheduler,
		num_epochs: int,
		log_wandb: bool = True,
		k_examples: int = 5
):
	# Evaluation Metrics
	bleu = evaluate.load("bleu")
	meteor = evaluate.load("meteor")
	rouge = evaluate.load("rouge")

	for epoch in range(num_epochs):
		# ---------------- Training Phase ----------------
		model.train()
		train_loss = 0
		for b, (images, captions) in enumerate(train_loader):		
			# Forward pass
			loss, cross_entropy_loss, contrastive_loss, cos_sim, input_ids, labels, logits = model(images, captions)
			train_loss += loss.item()
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			
			with torch.no_grad():
				pred_tokens = torch.argmax(logits, dim=-1) # [B, seq_len+1]
				if b == len(train_loader) - 1:
					print("-" * 50)
					print("\nTeacher Forcing Examples:\n")

					for i in range(min(k_examples, input_ids.size(0))):
						input_tokens = input_ids[i] # [seq_len+1]
						target_tokens = labels[i] # [seq_len+1]
						pred_tokens_seq = pred_tokens[i] # [seq_len+1]

						print(f"Example {i + 1}:\n")
						for j in range(len(target_tokens)):
							if target_tokens[j] == -100:
								continue

							input_text = model.decoder_tokenizer.decode(input_tokens[:j + 1], skip_special_tokens=False)
							pred_token = model.decoder_tokenizer.decode([pred_tokens_seq[j]], skip_special_tokens=False)
							target_token = model.decoder_tokenizer.decode([target_tokens[j]], skip_special_tokens=False)

							print(f"{input_text} [{pred_token}] ({target_token})")

						print("\n")
					print("-" * 50)

				print(f"Batch Loss: {loss.item():.4f}\t{b+1}/{len(train_loader)}")

				if log_wandb:
					# Log loss
					wandb.log({"train_batch_loss": loss.item()})
					wandb.log({"train_batch_loss(cross_entropy)": cross_entropy_loss.item()})
					wandb.log({"train_batch_loss(contrastive)": contrastive_loss.item()})

					# Log gradient magnitudes
					encoder_grad_norms = []
					proj_grad_norms = []
					decoder_grad_norms = []

					for name, param in model.named_parameters():
						if param.grad is not None:
							if "encoder" in name:
								encoder_grad_norms.append(param.grad.norm().item())
							elif "proj" in name:
								proj_grad_norms.append(param.grad.norm().item())
							elif "decoder" in name:
								decoder_grad_norms.append(param.grad.norm().item())

					if encoder_grad_norms:
						wandb.log({"grad_norm/encoder": sum(encoder_grad_norms) / len(encoder_grad_norms)})
					if proj_grad_norms:
						wandb.log({"grad_norm/proj": sum(proj_grad_norms) / len(proj_grad_norms)})
					if decoder_grad_norms:
						wandb.log({"grad_norm/decoder": sum(decoder_grad_norms) / len(decoder_grad_norms)})

					# Log learning rate
					for param_group in optimizer.param_groups:
						wandb.log({f"learning_rate/{param_group['name']}": param_group['lr']})

					# Log cosine similarity
					wandb.log({"image-text_cosine_sim": cos_sim})

					# Log token-level accuracy
					mask = labels != -100
					correct = (pred_tokens[mask] == labels[mask]).sum().item()
					total = mask.sum().item()
					token_accuracy = correct / total if total > 0 else 0.0
					wandb.log({"token_accuracy": token_accuracy})

		if scheduler:
			scheduler.step()

		# Evaluate training
		print("Evaluating training set...")
		model.eval()
		train_preds, train_gt = [], []
		with torch.no_grad():
			for images, captions in train_loader:			
				# Generate captions
				pred_captions = model(images, captions=None)
				train_preds.extend(pred_captions)
				train_gt.extend(captions)
		
		# Log distinct n-grams
		if log_wandb:
			wandb.log({"distinct_1-grams": distinct_ngrams(train_preds, n=1)})
			wandb.log({"distinct_2-grams": distinct_ngrams(train_preds, n=2)})

		# Print some random examples
		print("-"*50)
		print("\nTraining Examples:\n")
		for i in np.random.randint(0, len(train_preds), k_examples):
			print(f"Prediction: {train_preds[i]}")
			print(f"Groud Truth: {train_gt[i]}")
			print("\n")
		print("-"*50)

		# Compute Metrics
		train_bleu_1 = bleu.compute(predictions=train_preds, references=[[ref] for ref in train_gt], max_order=1)
		train_bleu_2 = bleu.compute(predictions=train_preds, references=[[ref] for ref in train_gt], max_order=2)
		train_meteor = meteor.compute(predictions=train_preds, references=[[ref] for ref in train_gt])
		train_rouge = rouge.compute(predictions=train_preds, references=[[ref] for ref in train_gt])

		# ---------------- Validation Phase ----------------
		print("Evaluating validation set...")
		model.eval()
		val_loss, val_ce_loss, val_con_loss, val_cos_sim, val_token_acc = 0, 0, 0, 0, 0
		val_preds, val_gt = [], []
		with torch.no_grad():
			for b, (images, captions) in enumerate(val_loader):			
				# Forward pass
				loss, cross_entropy_loss, contrastive_loss, cos_sim, input_ids, labels, logits = model(images, captions)
				val_loss += loss.item()
				val_ce_loss += cross_entropy_loss.item()
				val_con_loss += contrastive_loss.item()
				val_cos_sim += cos_sim

				pred_tokens = torch.argmax(logits, dim=-1) # [B, seq_len+1]
				if b == len(val_loader) - 1:
					print("-" * 50)
					print("\nTeacher Forcing Examples:\n")

					for i in range(min(k_examples, input_ids.size(0))):
						input_tokens = input_ids[i] # [seq_len+1]
						target_tokens = labels[i] # [seq_len+1]
						pred_tokens_seq = pred_tokens[i] # [seq_len+1]

						print(f"Example {i + 1}:\n")
						for j in range(len(target_tokens)):
							if target_tokens[j] == -100:
								continue

							input_text = model.decoder_tokenizer.decode(input_tokens[:j + 1], skip_special_tokens=False)
							pred_token = model.decoder_tokenizer.decode([pred_tokens_seq[j]], skip_special_tokens=False)
							target_token = model.decoder_tokenizer.decode([target_tokens[j]], skip_special_tokens=False)

							print(f"{input_text} [{pred_token}] ({target_token})")

						print("\n")
					print("-" * 50)

				# token-level accuracy
				mask = labels != -100
				correct = (pred_tokens[mask] == labels[mask]).sum().item()
				total = mask.sum().item()
				val_token_acc += correct / total if total > 0 else 0.0
				
				# Generate captions
				pred_captions = model(images, captions=None)
				val_preds.extend(pred_captions)
				val_gt.extend(captions)

		# Print some random examples
		print("-"*50)
		print("\nValidation Examples:\n")
		for i in np.random.randint(0, len(val_preds), k_examples):
			print(f"Prediction: {val_preds[i]}")
			print(f"Groud Truth: {val_gt[i]}")
			print("\n")
		print("-"*50)

		# Compute Metrics
		val_bleu_1 = bleu.compute(predictions=val_preds, references=[[ref] for ref in val_gt], max_order=1)
		val_bleu_2 = bleu.compute(predictions=val_preds, references=[[ref] for ref in val_gt], max_order=2)
		val_meteor = meteor.compute(predictions=val_preds, references=[[ref] for ref in val_gt])
		val_rouge = rouge.compute(predictions=val_preds, references=[[ref] for ref in val_gt])

		# Log Metrics
		avg_train_loss = train_loss / len(train_loader)
		avg_val_loss = val_loss / len(val_loader)
		avg_val_ce_loss = val_ce_loss / len(val_loader)
		avg_val_con_loss = val_con_loss / len(val_loader)
		avg_val_cos_sim = val_cos_sim / len(val_loader)
		avg_val_token_acc = val_token_acc / len(val_loader)

		print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
		if log_wandb:
			wandb.log({
				"epoch_train_loss": avg_train_loss,
				"epoch_val_loss": avg_val_loss,
				"epoch_val_loss(cross_entropy)": avg_val_ce_loss,
				"epoch_val_loss(contrastive)": avg_val_con_loss,
				"epoch_val_cosine_sim": avg_val_cos_sim,
				"epoch_val_token_acc": avg_val_token_acc,
				"BLEU-1": val_bleu_1["bleu"],
				"BLEU-2": val_bleu_2["bleu"],
				"ROUGE-L": val_rouge["rougeL"],
				"METEOR": val_meteor["meteor"],
				"train_BLEU-1": train_bleu_1["bleu"],
				"train_BLEU-2": train_bleu_2["bleu"],
				"train_ROUGE-L": train_rouge["rougeL"],
				"train_METEOR": train_meteor["meteor"]
			})
