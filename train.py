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
from src.ViTGptVED import ViTGptVED, train_ViTGptVED
from src.DinoGptVED import DinoGptVED, train_DinoGptVED
from src.DatasetCaption import FlickrDataset, ReceipesDataset, collate_fn_lst
from datetime import datetime

log_wandb = True

def train(
		config: dict,
		**kwargs
):
	# Initialize wandb
	model_name = config["model"]
	datetime_str = datetime.now().strftime("%d-%m_%H:%M:%S")
	save_name = f"{model_name}_{datetime_str}"
	if log_wandb:
		# join config with kwargs
		wandb_config = {**config, **kwargs}
		[wandb_config.pop(k) for k in ["out_dir", "data_dir", "gpu"]]
		wandb.init(
			project="CAP-GIA",
			name=save_name,
			config=wandb_config
		)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")

	# Prepare the dataset
	print("Preparing dataset...")
	data_path = os.path.join(kwargs["data_dir"], "flickr8k")
	train_set = FlickrDataset(data_path, split="train", data_size=kwargs["data_size"], return_img_path=True)
	val_set = FlickrDataset(data_path, split="val", data_size=kwargs["data_size"], return_img_path=True)
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
	elif model_name == "ViT-GPT-VED":
		model_class = ViTGptVED
		train_func = train_ViTGptVED
	elif model_name == "DINO-GPT-VED":
		model_class = DinoGptVED
		train_func = train_DinoGptVED
	output_dir = kwargs["out_dir"]
	model = model_class(
		output_dir=output_dir
	)
	model.to(device)

	# Prepare the training configuration
	print("Training model...")
	optimizer_name = config["optimizer"]
	if model_name.endswith("VED"):
		optimizer = getattr(torch.optim, optimizer_name)(
			model.parameters(),
			lr=config["lr"]
		)
	else:
		optimizer = getattr(torch.optim, optimizer_name)(
			[{"params": model.encoder.parameters(), "lr": config["lr"], "name": "encoder"},	
			{"params": model.proj.parameters(), "lr": config["lr"], "name": "proj"},
			{"params": model.decoder.parameters(), "lr": config["lr"], "name": "decoder"}],
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
		num_epochs=config["epochs"],
		log_wandb=log_wandb
	)

	# Save the model
	if kwargs["data_size"] == 1.0:
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		save_path = os.path.join(output_dir, f"{save_name}.pt")
		print(f"Saving model to {save_path}")
		torch.save(model.state_dict(), save_path)

	# Finish
	print("Done!")
	if log_wandb:
		wandb.finish()


if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_dir', type=str, default="out")
	parser.add_argument("--data_dir", type=str, default="")
	parser.add_argument("--gpu", type=int, default=-1)
	parser.add_argument("--data_size", type=float, default=0.05)
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
		config=config,
		**vars(args)
	)
