import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, VisionEncoderDecoderModel, GPT2TokenizerFast, GenerationConfig
import evaluate
from typing import Literal


def distinct_ngrams(predictions, n=1):
	ngrams = set()
	for pred in predictions:
		tokens = pred.split()
		ngrams.update(zip(*[tokens[i:] for i in range(n)]))
	return len(ngrams) / sum(len(pred.split()) for pred in predictions)


class DinoGptVED(nn.Module):
	def __init__(
			self,
			output_dir: str
	):
		super().__init__()

		# Initialize VED model with pretrained DINO and GPT-2
		self.VED = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
			"google/vit-base-patch16-224-in21k", "gpt2", cache_dir=output_dir
		)
		self.encoder_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=output_dir)
		self.decoder_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=output_dir)
		
		# Add special tokens
		if self.decoder_tokenizer.bos_token is None:
			self.decoder_tokenizer.add_special_tokens({'bos_token': '<|startoftext|>'})
		if self.decoder_tokenizer.pad_token is None:
			self.decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		if self.decoder_tokenizer.eos_token is None:
			self.decoder_tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

		self.VED.config.bos_token_id = self.decoder_tokenizer.bos_token_id
		self.VED.config.pad_token_id = self.decoder_tokenizer.pad_token_id
		self.VED.config.eos_token_id = self.decoder_tokenizer.eos_token_id
		self.VED.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
		self.VED.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

	def forward(
			self,
			images: list, # PIL images
			captions: list[str] = None,
			max_length: int=30,
			temperature: float=0.7,
			repetition_penalty: float=1.2,
			length_penalty: float=0.0,
			inference_mode: Literal["sampling", "beam_search"] = "sampling",
			n_beams: int=5,
			top_k: int=50,
			top_p: float=0.95
	):
		device = next(self.parameters()).device

		# Encode images with DINO
		pixel_values = self.encoder_processor(images, return_tensors="pt").pixel_values.to(device)

		if captions is not None:
			# Training/Validation Mode

			# Encode captions with GPT-2
			encodings = self.decoder_tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
			labels = encodings.input_ids.to(device)
			attention_mask = encodings.attention_mask.to(device)
			labels[attention_mask == 0] = -100

			# Compute loss
			loss = self.VED(
				pixel_values=pixel_values,
				labels=labels
			).loss
			return loss
		
		else:
			# Inference Mode
			# https://huggingface.co/docs/transformers/main_classes/text_generation
			generation_config = GenerationConfig(
				max_length=max_length,
				temperature=temperature,
				repetition_penalty=repetition_penalty,
				length_penalty=length_penalty if n_beams > 1 else None,
				do_sample=inference_mode == "sampling",
				num_beams=n_beams if inference_mode == "beam_search" else 1,
				top_k=top_k,
				top_p=top_p,
				pad_token_id=self.decoder_tokenizer.pad_token_id,
				eos_token_id=self.decoder_tokenizer.eos_token_id,
			)
			encodings = self.decoder_tokenizer([""], return_tensors='pt')  # Empty input or BOS
			attention_mask = encodings.attention_mask.to(device)
			outputs = self.VED.generate(
				pixel_values,
				generation_config=generation_config,
				attention_mask=attention_mask
			)
			generated_texts = self.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
			return generated_texts
		

def train_DinoGptVED(
		model: DinoGptVED,
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
			loss = model(images, captions)
			train_loss += loss.item()
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			with torch.no_grad():
				print(f"Batch Loss: {loss.item():.4f}\t{b+1}/{len(train_loader)}")
				if log_wandb:
					# Log loss
					wandb.log({"train_batch_loss": loss.item()})

					# Log gradient magnitudes
					grad_norms = []
					for p in model.parameters():
						if p.grad is not None:
							grad_norms.append(p.grad.norm().item())
					wandb.log({"grad_norm": np.mean(grad_norms)})

					# Log learning rate
					wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

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
		val_loss = 0
		val_preds, val_gt = [], []
		with torch.no_grad():
			for b, (images, captions) in enumerate(val_loader):			
				# Forward pass
				loss = model(images, captions)
				val_loss += loss.item()

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

		print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
		if log_wandb:
			wandb.log({
				"epoch_train_loss": avg_train_loss,
				"epoch_val_loss": avg_val_loss,
				"BLEU-1": val_bleu_1["bleu"],
				"BLEU-2": val_bleu_2["bleu"],
				"ROUGE-L": val_rouge["rougeL"],
				"METEOR": val_meteor["meteor"],
				"train_BLEU-1": train_bleu_1["bleu"],
				"train_BLEU-2": train_bleu_2["bleu"],
				"train_ROUGE-L": train_rouge["rougeL"],
				"train_METEOR": train_meteor["meteor"]
			})
