import wandb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch.utils.data import DataLoader
from src.DatasetCaption import explode_caption_lst
from transformers import (
	AutoImageProcessor,
	VisionEncoderDecoderModel,
	VisionEncoderDecoderConfig,
	GPT2TokenizerFast,
	GenerationConfig,
	ViTConfig,
	ViTModel
)
import evaluate
from typing import Literal
# from peft import LoraConfig, get_peft_model, TaskType#, EvaConfig, initialize_lora_eva_weights

def distinct_ngrams(predictions, n=1):
	ngrams = set()
	for pred in predictions:
		tokens = pred.split()
		ngrams.update(zip(*[tokens[i:] for i in range(n)]))
	return len(ngrams) / sum(len(pred.split()) for pred in predictions)

def custom_init(module):
	if isinstance(module, nn.Linear):
		xavier_uniform_(module.weight)
		if module.bias is not None:
			zeros_(module.bias)
	elif isinstance(module, nn.LayerNorm):
		nn.init.ones_(module.weight)
		nn.init.zeros_(module.bias)

class ViTGptVED(nn.Module):
	def __init__(
			self,
			output_dir: str
	):
		super().__init__()

		# Initialize VED model with pretrained ViT and GPT-2
		decoder = VisionEncoderDecoderModel.from_pretrained(
			"nlpconnect/vit-gpt2-image-captioning", cache_dir=output_dir
		).decoder
		encoder_config = ViTConfig(
			hidden_size=1024,
			num_hidden_layers=24,
			num_attention_heads=16,
		)
		encoder = ViTModel(encoder_config)
		VED_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
		self.VED = VisionEncoderDecoderModel(config=VED_config, encoder=encoder, decoder=decoder)
		self.VED.encoder.apply(custom_init)
		self.encoder_processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=output_dir)
		self.decoder_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=output_dir)

		# Add special tokens
		self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
		self.VED.config.pad_token_id = self.decoder_tokenizer.pad_token_id
		self.VED.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id

		self.contrastive_criterion = nn.CosineEmbeddingLoss()

		# LoRA
		# def print_module_names(model):
		# 	for name, _ in model.named_parameters():
		# 		print(name)
		# print_module_names(self.VED)
		# lora_config = LoraConfig(
		# 	r=16,
		# 	lora_alpha=32,
		# 	target_modules=r".*\.attention\.|.*\.c_attn|.*\.c_proj|.*\.mlp\.",
		# 	lora_dropout=0.1,
		# 	task_type=TaskType.SEQ_2_SEQ_LM,
		# 	init_lora_weights="gaussian",
		# 	# eva_config = EvaConfig(rho = 2.0),
		# )
		# self.VED = get_peft_model(
		# 	model=self.VED,
		# 	peft_config=lora_config
		# )

	def image_text_contrastive_loss_baseline(self, image_feat, text_feat, temperature=0.07):
		N = image_feat.shape[0]
		logits = torch.matmul(image_feat, text_feat.t())
		logits /= temperature
		gt = torch.arange(N, device=logits.device)
		loss1 = torch.nn.functional.cross_entropy(logits, gt)
		loss2 = torch.nn.functional.cross_entropy(logits.t(), gt)
		return (loss1 + loss2) / 2

	def forward(
			self,
			images: list, # PIL images
			captions: list[str] = None,
			max_new_tokens: int=20,
			temperature: float=0.7,
			repetition_penalty: float=4.0,
			length_penalty: float=3.0,
			lambda_contrastive: float=10.0, # Contrastive loss weight
			inference_mode: Literal["sampling", "beam_search"] = "beam_search",
			n_beams: int=5,
			top_k: int=50,
			top_p: float=0.95
	):
		device = next(self.parameters()).device

		# Encode images with ViT
		pixel_values = self.encoder_processor(images, return_tensors="pt").pixel_values.to(device)

		if captions is not None:
			# Training/Validation Mode

			# Encode captions with GPT-2
			encodings = self.decoder_tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
			labels = encodings.input_ids.to(device)
			attention_mask = encodings.attention_mask.to(device)
			labels[attention_mask == 0] = -100

			# Compute cross-entropy loss
			encoder_outputs = self.VED.encoder(pixel_values)
			outputs = self.VED(
				encoder_outputs=encoder_outputs,
				labels=labels,
				decoder_attention_mask=attention_mask,
				output_hidden_states=True
			)
			ce_loss = outputs.loss

			# Compute contrastive loss
			encoder_hidden_states = outputs.encoder_last_hidden_state
			decoder_hidden_states = outputs.decoder_hidden_states[-1]
			image_latent = F.normalize(encoder_hidden_states.mean(dim=1), p=2, dim=1)
			text_latent = F.normalize(decoder_hidden_states.mean(dim=1), p=2, dim=1)
			contrastive_loss = self.image_text_contrastive_loss_baseline(image_latent, text_latent)
			total_loss = ce_loss + lambda_contrastive * contrastive_loss
			with torch.no_grad():
				cos_sim = F.cosine_similarity(image_latent, text_latent, dim=1).mean().item()
			return total_loss, ce_loss, contrastive_loss, cos_sim

		else:
			# Inference Mode
			# https://huggingface.co/docs/transformers/main_classes/text_generation
			generation_config = GenerationConfig(
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				repetition_penalty=repetition_penalty,
				length_penalty=length_penalty if (n_beams > 1 and inference_mode == "beam_search") else None,
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
				pixel_values=pixel_values,
				generation_config=generation_config
			)
			generated_texts = self.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
			return generated_texts
		

def train_ViTGptVED(
		model: ViTGptVED,
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
		for b, (images, captions, _) in enumerate(train_loader):		
			# Forward pass
			loss, ce_loss, contrastive_loss, cos_sim = model(images, captions)
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
					wandb.log({"train_batch_loss(cross_entropy)": ce_loss.item()})
					wandb.log({"train_batch_loss(contrastive)": contrastive_loss.item()})

					# Log gradient magnitudes
					grad_norms = []
					encoder_grad_norms = []
					proj_grad_norms = []
					decoder_grad_norms = []
					for name, param in model.named_parameters():
						if param.grad is not None:
							grad_norms.append(param.grad.norm().item())
							if "encoder" in name:
								encoder_grad_norms.append(param.grad.norm().item())
							elif "proj" in name:
								proj_grad_norms.append(param.grad.norm().item())
							elif "decoder" in name:
								decoder_grad_norms.append(param.grad.norm().item())
					wandb.log({"grad_norm": np.mean(grad_norms)})
					if encoder_grad_norms:
						wandb.log({"grad_norm/encoder": sum(encoder_grad_norms) / len(encoder_grad_norms)})
					if proj_grad_norms:
						wandb.log({"grad_norm/proj": sum(proj_grad_norms) / len(proj_grad_norms)})
					if decoder_grad_norms:
						wandb.log({"grad_norm/decoder": sum(decoder_grad_norms) / len(decoder_grad_norms)})

					# Log learning rate
					wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

					# Log cosine similarity
					wandb.log({"image-text_cosine_sim": cos_sim})

			if scheduler:
				scheduler.step()

		# Evaluate training
		print("Evaluating training set...")
		model.eval()
		train_preds, train_gt, train_img_paths = [], [], []
		with torch.no_grad():
			for images, captions, img_paths in train_loader:			
				# Generate captions
				pred_captions = model(images, captions=None)
				train_preds.extend(pred_captions)
				train_gt.extend(captions)
				train_img_paths.extend(img_paths)
		
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
			print(f"Image Path: {train_img_paths[i]}")
			print("\n")
		print("-"*50)

		# Compute Metrics
		if not isinstance(train_gt[0], list):
			train_gt = [[ref] for ref in train_gt]
		train_bleu_1 = bleu.compute(predictions=train_preds, references=train_gt, max_order=1)
		train_bleu_2 = bleu.compute(predictions=train_preds, references=train_gt, max_order=2)
		train_meteor = meteor.compute(predictions=train_preds, references=train_gt)
		train_rouge = rouge.compute(predictions=train_preds, references=train_gt)

		# ---------------- Validation Phase ----------------
		print("Evaluating validation set...")
		model.eval()
		val_loss, val_ce_loss, val_con_loss, val_cos_sim = 0, 0, 0, 0
		val_preds, val_gt, val_img_paths = [], [], []
		with torch.no_grad():
			for b, (images, captions, img_paths) in enumerate(val_loader):		
				# Forward pass
				images_exp, captions_exp = explode_caption_lst(images, captions)	
				loss, ce_loss, contrastive_loss, cos_sim = model(images_exp, captions_exp)
				val_loss += loss.item()
				val_ce_loss += ce_loss.item()
				val_con_loss += contrastive_loss.item()
				val_cos_sim += cos_sim

				# Generate captions
				pred_captions = model(images, captions=None)
				val_preds.extend(pred_captions)
				val_gt.extend(captions)
				val_img_paths.extend(img_paths)

		# Print some random examples
		print("-"*50)
		print("\nValidation Examples:\n")
		for i in np.random.randint(0, len(val_preds), k_examples):
			print(f"Prediction: {val_preds[i]}")
			print(f"Groud Truth: {val_gt[i]}")
			print(f"Image Path: {val_img_paths[i]}")
			print("\n")
		print("-"*50)

		# Compute Metrics
		if not isinstance(val_gt[0], list):
			val_gt = [[ref] for ref in val_gt]
		val_bleu_1 = bleu.compute(predictions=val_preds, references=val_gt, max_order=1)
		val_bleu_2 = bleu.compute(predictions=val_preds, references=val_gt, max_order=2)
		val_meteor = meteor.compute(predictions=val_preds, references=val_gt)
		val_rouge = rouge.compute(predictions=val_preds, references=val_gt)

		# Log Metrics
		avg_train_loss = train_loss / len(train_loader)
		avg_val_loss = val_loss / len(val_loader)
		avg_val_ce_loss = val_ce_loss / len(val_loader)
		avg_val_con_loss = val_con_loss / len(val_loader)
		avg_val_cos_sim = val_cos_sim / len(val_loader)

		print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
		if log_wandb:
			wandb.log({
				"epoch_train_loss": avg_train_loss,
				"epoch_val_loss": avg_val_loss,
				"epoch_val_loss(cross_entropy)": avg_val_ce_loss,
				"epoch_val_loss(contrastive)": avg_val_con_loss,
				"epoch_val_cosine_sim": avg_val_cos_sim,
				"BLEU-1": val_bleu_1["bleu"],
				"BLEU-2": val_bleu_2["bleu"],
				"ROUGE-L": val_rouge["rougeL"],
				"METEOR": val_meteor["meteor"],
				"train_BLEU-1": train_bleu_1["bleu"],
				"train_BLEU-2": train_bleu_2["bleu"],
				"train_ROUGE-L": train_rouge["rougeL"],
				"train_METEOR": train_meteor["meteor"]
			})
