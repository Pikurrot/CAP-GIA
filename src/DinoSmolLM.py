import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
import evaluate

class DinoSmolLM(nn.Module):
	def __init__(
			self,
			output_dir: str
	):
		super(DinoSmolLM, self).__init__()
		self.output_dir = output_dir

		# Load pre-trained DINO model
		self.encoder_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", cache_dir=self.output_dir)
		self.encoder = AutoModel.from_pretrained("facebook/dinov2-small", cache_dir=self.output_dir)

		# Load pre-trained SmolLM model
		self.decoder_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", cache_dir=self.output_dir)
		self.decoder = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", cache_dir=self.output_dir)
		special_tokens = {'bos_token': '<start>', 'eos_token': '<end>', 'pad_token': '[PAD]'}
		self.decoder_tokenizer.add_special_tokens(special_tokens)
		self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

		# Freeze DINO parameters
		# for param in self.encoder.parameters():
		# 	param.requires_grad = False
		# for layer in self.encoder.encoder.layer[-2:]:
		# 	for param in layer.parameters():
		# 		param.requires_grad = True

		# Linear projection layer
		self.proj = nn.Sequential(
			nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size),
			nn.ReLU(),
			nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
		)

	def forward(
			self,
			images: torch.Tensor, # (bs, channels, h, w)
			captions: torch.Tensor = None, # (bs, seq_len)
			max_seq_len: int = 20
	):
		# Extract image features
		inputs = self.encoder_processor(images, return_tensors="pt")
		inputs = inputs.to(self.encoder.device)
		image_features = self.encoder(**inputs)[0][:, 0, :] # (bs, hidden_size)

		# Project image features to decoder embedding size
		image_embeddings = self.proj(image_features).unsqueeze(1) # (bs, 1, n_embd)

		if captions is not None:
			# Training with teacher forcing
			captions = [f"<start> {caption.lower().strip()} <end>" for caption in captions]
			encoding = self.decoder_tokenizer(
				captions,
				return_tensors="pt",
				padding=True,
				truncation=True,
				add_special_tokens=True
			)
			input_ids = encoding.input_ids.to(self.encoder.device)  # (bs, seq_len)
			attention_mask = encoding.attention_mask.to(self.encoder.device)  # (bs, seq_len)

			# Get input embeddings
			inputs_embeds = self.decoder.model.embed_tokens(input_ids)  # (bs, seq_len, n_embd)

			# Concatenate image embeddings with inputs
			inputs_embeds = torch.cat((image_embeddings, inputs_embeds[:, :-1, :]), dim=1)  # (bs, seq_len + 1, n_embd)

			# Update attention mask
			attention_mask = torch.cat(
				(torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype),
				attention_mask[:, :-1]),
				dim=1
			)  # (bs, seq_len + 1)

			# Labels
			labels = input_ids

			# Compute loss
			outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
			loss = outputs.loss
			return loss
		else:
			# Inference
			batch_size = image_embeddings.size(0)
			start_token_id = self.decoder_tokenizer.bos_token_id
			start_token_embedding = self.decoder.model.embed_tokens(
				torch.tensor([start_token_id], device=image_embeddings.device)
			).unsqueeze(0)  # (1, 1, n_embd)
			start_token_embeddings = start_token_embedding.repeat(batch_size, 1, 1)  # (bs, 1, n_embd)

			# Concatenate image embeddings and start token embedding
			inputs_embeds = torch.cat((image_embeddings, start_token_embeddings), dim=1)  # (bs, 2, n_embd)

			# Update attention mask
			attention_mask = torch.ones((batch_size, 2), dtype=torch.long, device=image_embeddings.device)  # (bs, 2)

			# Generate captions
			generated_ids = self.decoder.generate(
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				max_new_tokens=max_seq_len,
				num_beams=5,
				repetition_penalty=3.0,
				do_sample=True,
				top_k=50,
				top_p=0.9,
				early_stopping=True,
				pad_token_id=self.decoder_tokenizer.pad_token_id,
				eos_token_id=self.decoder_tokenizer.eos_token_id
			)

			# Remove image and <start> token ids
			generated_ids = generated_ids[:, 2:]  # (bs, generated_seq_len)

			# Decode the generated tokens
			captions = self.decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

			return captions


def train_DinoSmolLM(
		model: DinoSmolLM,
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
