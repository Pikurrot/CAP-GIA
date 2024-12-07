import argparse
import os
import wandb
import yaml
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from src.DinoGpt import DinoGpt, train_DinoGpt
from src.DinoSmolLM import DinoSmolLM, train_DinoSmolLM
from src.ViTGpt import ViTGpt, train_ViTGpt
from src.DatasetCaption import ReceipesDataset, collate_fn_lst
from datetime import datetime

log_wandb = True

def train(
		output_dir: str,
		data_dir: str,
		config: dict
):
	# Initialize wandb
	model_name = config["model"]
	datetime_str = datetime.now().strftime("%d-%m_%H:%M:%S")
	save_name = f"{model_name}_{datetime_str}"
	if log_wandb:
		wandb.init(
			project="CAP-GIA",
			name=save_name,
			config=config
		)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")

	# Prepare the dataset
	print("Preparing dataset...")
	data_path = os.path.join(data_dir, "receipes")
	train_set = ReceipesDataset(data_path, split="train")
	val_set = ReceipesDataset(data_path, split="val")
	train_loader = DataLoader(
		train_set,
		batch_size=config["batch_size"],
		shuffle=True,
		collate_fn=collate_fn_lst
	)
	val_loader = DataLoader(
		val_set,
		batch_size=config["batch_size"],
		shuffle=False,
		collate_fn=collate_fn_lst
	)

	# Prepare the model
	print(f"Preparing model {model_name}...")
	if model_name == "ViT-GPT":
		model_class = ViTGpt
		train_func = train_ViTGpt
	elif model_name == "DINO-GPT":
		model_class = DinoGpt
		train_func = train_DinoGpt
	elif model_name == "DINO-SmolLM":
		model_class = DinoSmolLM
		train_func = train_DinoSmolLM
	model = model_class(
		output_dir=output_dir
	)
	model.to(device)

	# Prepare the training configuration
	print("Training model...")
	optimizer_name = config["optimizer"]
	optimizer = getattr(torch.optim, optimizer_name)(
		[{"params": model.encoder.parameters(), "lr": 1e-4},
		{"params": model.proj.parameters(), "lr": 1e-4},
		{"params": model.decoder.parameters(), "lr": 1e-4},]
	)
	scheduler_conf = config["scheduler"]
	if scheduler_conf is not None:
		scheduler_name = scheduler_conf["name"]
		scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
			optimizer,
			**{k: v for k, v in scheduler_conf.items() if k != "name"}
		)
	else:
		scheduler = None

	# Train the model
	train_func(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		optimizer=optimizer,
		scheduler=scheduler,
		device=device,
		num_epochs=config["epochs"],
		log_wandb=log_wandb
	)

	# Save the model
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	save_path = os.path.join(output_dir, f"{save_name}.pt")
	print(f"Saving model to {save_path}")
	torch.save(model.state_dict(), save_path)
	# if log_wandb:
	# 	wandb.save(save_path)

	# Finish
	print("Done!")
	if log_wandb:
		wandb.finish()


if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_dir', type=str, default='out')
	parser.add_argument("--data_dir", type=str, default="")
	parser.add_argument("--gpu", type=int, default=-1)
	args = parser.parse_args()
	print(args)

	# Load config
	load_dotenv()
	wandb_key = os.getenv("WANDB_KEY")
	with open("config/train.yml", "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# Train model on dataset
	if args.gpu >= 0:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	train(
		output_dir=args.out_dir,
		data_dir=args.data_dir,
		config=config
	)
