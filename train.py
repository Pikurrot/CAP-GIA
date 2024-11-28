import argparse
import os
import wandb
import yaml
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from src.ResnetCaption import ResnetLSTMCaption, train_resnetLSTMCaption
from src.DatasetCaption import FlickrDataset, collate_fn

log_wandb = True

def train(
		model_name: str,
		dataset_name: str,
		output_dir: str,
		data_dir: str,
		config: dict
):
	# Initialize wandb
	if log_wandb:
		wandb.init(
			project="CAP-GIA",
			name=f"{model_name}-{dataset_name}",
			config=config
		)

	# Prepare the dataset
	print(f"Preparing dataset {dataset_name}...")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	data_path = os.path.join(data_dir, dataset_name)
	if dataset_name.startswith("flickr"):
		train_set = FlickrDataset(
			data_path=data_path,
			split="train"
		)
		val_set = FlickrDataset(
			data_path=data_path,
			split="val"
		)
	print(f"Vocabulary size: {train_set.vocab_size}")
	train_loader = DataLoader(
		train_set,
		batch_size=config["batch_size"],
		shuffle=True,
		collate_fn=lambda x: collate_fn(x, train_set.word2idx)
	)
	val_loader = DataLoader(
		val_set,
		batch_size=config["batch_size"],
		shuffle=False,
		collate_fn=lambda x: collate_fn(x, train_set.word2idx)
	)

	# Prepare the model
	print(f"Preparing model {model_name}...")
	if model_name.startswith("resnet"):
		model = ResnetLSTMCaption(
			resnet_name=model_name,
			output_dir=output_dir,
			hidden_size=config["hidden_size"],
			vocab_size=train_set.vocab_size,
			embedding_dim=config["embedding_dim"],
			num_layers=config["num_layers"]
		)
		model.to(device)
		train_model = train_resnetLSTMCaption

	# Prepare the training configuration
	print("Training model...")
	optimizer_name = config["optimizer"]
	optimizer = getattr(torch.optim, optimizer_name)(
		model.parameters(),
		lr=config["lr"]
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
	criterion = torch.nn.CrossEntropyLoss(ignore_index=train_set.word2idx['<pad>'])

	# Train the model
	train_model(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		idx2word=train_set.idx2word,
		optimizer=optimizer,
		scheduler=scheduler,
		criterion=criterion,
		device=device,
		num_epochs=config["epochs"],
		log_wandb=log_wandb
	)

	# Save the model
	print("Saving model...")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	save_path = os.path.join(output_dir, f"{model_name}-{dataset_name}.pt")
	i = 0
	while os.path.exists(save_path):
		save_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{i}.pt")
		i += 1
	torch.save(model.state_dict(), save_path)
	if log_wandb:
		wandb.save(save_path)

	# Finish
	print("Done!")
	if log_wandb:
		wandb.finish()


if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='resnet-18')
	parser.add_argument('--dataset', type=str, default='flickr-8k')
	parser.add_argument('--out_dir', type=str, default='out')
	parser.add_argument("--data_dir", type=str, default="")
	args = parser.parse_args()
	print(args)
	
	assert args.model in ["resnet-18", "resnet-34", "resnet-50"]
	assert args.dataset in ["flickr-8k", "flickr-30k"]

	# Load config
	load_dotenv()
	wandb_key = os.getenv("WANDB_KEY")
	with open("config/train.yml", "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	assert args.model in config, f"Model {args.model} not found in config"
	assert args.dataset in config[args.model], f"Dataset {args.dataset} not found in config[{args.model}]"
	model_dataset_config = config[args.model][args.dataset]

	# Train model on dataset
	train(
		model_name=args.model,
		dataset_name=args.dataset,
		output_dir=args.out_dir,
		data_dir=args.data_dir,
		config=model_dataset_config
	)
