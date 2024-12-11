import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Literal, Union

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						 std=[0.229, 0.224, 0.225])
])

def destransform(image: torch.Tensor):
	image = image.permute(1, 2, 0).numpy()
	image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
	image = np.clip(image, 0, 1)
	image = image[..., ::-1]
	return image

def explode_caption_df(
		df: pd.DataFrame
):
	return df.explode("caption").reset_index(drop=True)

def explode_caption_lst(
		col1: list,
		col2: list
):
	df = pd.DataFrame({"image": col1, "caption": col2})
	res_df = explode_caption_df(df)
	res_col1 = res_df["image"].tolist()
	res_col2 = res_df["caption"].tolist()
	return res_col1, res_col2

class FlickrDataset(Dataset):
	def __init__(
			self,
			data_path: str,
			transform_image: bool = False,
			split: Literal["train", "val", "test"] = "train",
			split_size: list = [0.7, 0.1, 0.2],
			data_size: int=1.0
	):
		super(FlickrDataset, self).__init__()
		self.img_path = os.path.join(data_path, 'Images')
		self.cap_path = os.path.join(data_path, 'captions.txt')
		self.cap_data = pd.read_csv(self.cap_path)
		self.transform_image = transform_image
		self.split = split

		# Create vocabulary
		# self.vocab = set()
		# for caption in self.cap_data['caption']:
		# 	self.vocab.update(caption.split())
		# self.special_tokens = ['<start>', '<end>', '<pad>', '<unk>']
		# self.vocab.update(self.special_tokens)
		# self.vocab = sorted(self.vocab)
		# self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
		# self.idx2word = {idx: word for word, idx in self.word2idx.items()}
		# self.vocab_size = len(self.vocab)

		# Split data
		self.cap_data = self.cap_data.groupby("image").agg({"caption": list}).sort_index().reset_index()
		total_size = len(self.cap_data)
		train_end = int(split_size[0] * total_size)
		val_end = train_end + int(split_size[1] * total_size)

		if split == "train":
			self.cap_data = self.cap_data[:train_end]
		elif split == "val":
			self.cap_data = self.cap_data[train_end:val_end]
		elif split == "test":
			self.cap_data = self.cap_data[val_end:]

		if split == "train": # Undo grouping
			self.cap_data = explode_caption_df(self.cap_data)
		self.cap_data = self.cap_data.sample(frac=data_size, random_state=42)

	def __len__(self):
		return len(self.cap_data)
	
	def __getitem__(self, idx):
		img_name = os.path.join(self.img_path, self.cap_data.iloc[idx, 0])
		image = Image.open(img_name).convert('RGB')
		# image = np.array(image)#[..., ::-1]
		# image = Image.fromarray(image)
		if self.transform_image:
			image = transform(image)
		caption = self.cap_data.iloc[idx, 1]
		return image, caption


class ReceipesDataset(Dataset):
	def __init__(
			self,
			data_path: str,
			transform_image: bool = False,
			split: Literal["train", "val", "test"] = "train",
			split_size: list = [0.7, 0.1, 0.2],
			data_size: int=1.0
	):
		super(ReceipesDataset, self).__init__()
		self.img_path = os.path.join(data_path, 'FoodImages', 'Food Images')
		self.cap_path = os.path.join(data_path, 'FoodIngredientsAndReceipesDatasetWithImageNameMapping.csv')
		self.cap_data = pd.read_csv(self.cap_path)
		self.transform_image = transform_image
		self.split = split

		# Clean data
		self.cap_data = self.cap_data.dropna(subset=["Title"]) # TODO: Add other columns
		self.cap_data = self.cap_data[self.cap_data["Title"].apply(lambda x: len(x.split()) > 0)]
		self.cap_data = self.cap_data[self.cap_data["Image_Name"].apply(lambda x: x != "#NAME?")]

		# Create vocabulary
		# self.vocab = set()
		# for caption in self.cap_data["Title"]: # TODO: Add other columns
		# 	self.vocab.update(caption.split())
		# self.special_tokens = ['<start>', '<end>', '<pad>', '<unk>']
		# self.vocab.update(self.special_tokens)
		# self.vocab = sorted(self.vocab)
		# self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
		# self.idx2word = {idx: word for word, idx in self.word2idx.items()}
		# self.vocab_size = len(self.vocab)

		# Split data
		total_size = len(self.cap_data)
		train_end = int(split_size[0] * total_size)
		val_end = train_end + int(split_size[1] * total_size)

		if split == "train":
			self.cap_data = self.cap_data[:train_end]
		elif split == "val":
			self.cap_data = self.cap_data[train_end:val_end]
		elif split == "test":
			self.cap_data = self.cap_data[val_end:]

		self.cap_data = self.cap_data.sample(frac=data_size, random_state=42)
	
	def __len__(self):
		return len(self.cap_data)
	
	def verify_paths(self):
		for idx in range(len(self)):
			img_name = os.path.join(self.img_path, self.cap_data.iloc[idx, 4]) + ".jpg"
			if not os.path.exists(img_name):
				print(f"Image {img_name} does not exist.")
	
	def __getitem__(self, idx):
		img_name = os.path.join(self.img_path, self.cap_data.iloc[idx, 4]) + ".jpg"
		image = Image.open(img_name).convert('RGB')
		# image = np.array(image)[..., ::-1]
		# image = Image.fromarray(image)
		if self.transform_image:
			image = transform(image)
		caption = self.cap_data.iloc[idx, 1]
		return image, caption


def collate_fn_tensor(batch, word2idx):
	images, captions = zip(*batch)
	images = torch.stack(images, dim=0)
	captions = [
		torch.tensor(
			[word2idx['<start>']] + 
			[word2idx.get(word, word2idx['<unk>']) for word in caption.split()] + 
			[word2idx['<end>']]
		)
		for caption in captions
	]
	captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=word2idx['<pad>'])
	return images, captions


def collate_fn_lst(batch):
	# images come in PIL format
	images, captions = zip(*batch)
	return images, captions


if __name__ == '__main__':
	data_path = "/media/eric/D/datasets/receipes"
	dataset = ReceipesDataset(data_path, transform_image=False, split="train")
	print(len(dataset))
	image, caption = dataset[100] # image is PIL without transform
	cv2.imshow('image', np.array(image))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_lst)
	for images, captions in dataloader:
		print(len(images), len(captions))
		break
	
	dataset.verify_paths()
	print("Done.")
