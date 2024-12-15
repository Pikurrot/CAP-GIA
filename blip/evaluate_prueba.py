from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Literal 
import os 
import pandas as pd 
from torchvision import transforms
import evaluate


# Clase ReceipesDataset
class ReceipesDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform_image: bool = False,
        split: Literal["train", "val", "test"] = "train",
        split_size: list = [0.7, 0.1, 0.2],
        data_size: float = 1.0,
        processor = AutoProcessor.from_pretrained("model_resources")
    

    ):
        super(ReceipesDataset, self).__init__()
        self.img_path = os.path.join(data_path, "FoodImages", "Food Images")
        self.cap_path = os.path.join(
            data_path, "FoodIngredientsAndReceipesDatasetWithImageNameMapping.csv"
        )
        self.cap_data = pd.read_csv(self.cap_path)
        self.transform_image = transform_image
        self.split = split
        self.processor = processor 
        
        # Limpieza de datos
        self.cap_data = self.cap_data.dropna(subset=["Title"])
        self.cap_data = self.cap_data[
            self.cap_data["Title"].apply(lambda x: len(x.split()) > 0)
        ]
        self.cap_data = self.cap_data[
            self.cap_data["Image_Name"].apply(lambda x: x != "#NAME?")
        ]

        # División de datos
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

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.cap_data.iloc[idx]["Image_Name"])
        img_name += ".jpg"
        image = Image.open(img_name).convert("RGB")
        if self.transform_image:
            image = transform(image)
        caption = self.cap_data.iloc[idx]["Title"]
        encoding = self.processor(images = image , text = caption, padding= "max_length", return_tensors= "pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

transform = transforms.Compose([
	transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the pre-trained model from the checkpoint 5 which is the best checkpoint
model_finetuned = AutoModelForVision2Seq.from_pretrained("model_resources")
model_finetuned.to(device)

checkpoint_path = "/home/ldomene/CAP-GIA/blip/checkpoints/epoch_5.pth"
state_dict = torch.load(checkpoint_path, map_location=device,weights_only=True)
model_finetuned.load_state_dict(state_dict, strict=False)


bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

model_finetuned.eval()
predictions = []
references = []
images = []

# Load the test dataset
data_path = "/home/ldomene/caption_data/receipes"
test_dataset = ReceipesDataset(data_path=data_path, transform_image=False, split="test")
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)

processor = AutoProcessor.from_pretrained("model_resources")

# Generate Predictions
for batch in test_loader:
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["pixel_values"].to(device)
        
        generated_output = model_finetuned.generate(
            pixel_values=image, max_new_tokens=64
        )
        predictions.extend(processor.batch_decode(generated_output, skip_special_tokens=True))
        references.extend(processor.batch_decode(input_ids, skip_special_tokens=True))
        images.extend(image.cpu())
        
res_bleu_1 = bleu.compute(
    predictions=predictions, references=[[ref] for ref in references], max_order=1
)
res_bleu_2 = bleu.compute(
    predictions=predictions, references=[[ref] for ref in references], max_order=2
)
res_meteor = meteor.compute(
    predictions=predictions, references=[[ref] for ref in references]
)
res_rouge = rouge.compute(
    predictions=predictions, references=[[ref] for ref in references]
)

# Plot the results and save them
print(
    f"BLEU-1 = {res_bleu_1['bleu']:.4f}, BLEU-2 = {res_bleu_2['bleu']:.4f}, METEOR = {res_meteor['meteor']:.4f}, ROUGE-L = {res_rouge['rougeL']:.4f}"
)

import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import random

def save_example(image_tensor, prediction, reference, score, idx, output_dir):
    image = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    
    image = image.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(12, 12))  # Aumentar tamaño de la figura
    plt.imshow(image, interpolation='bilinear')  # Usar interpolación para suavizar
    plt.axis('off')
    plt.title(
        f'METEOR Score: {score:.4f}\nPrediction: {prediction}\nGround Truth: {reference}', 
        wrap=True
    )

    plt.savefig(os.path.join(output_dir, f'example_{idx}.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

# After computing predictions
example_scores = []
for pred, ref in zip(predictions, references):
    score = meteor.compute(predictions=[pred], references=[[ref]])['meteor']
    example_scores.append(score)

# Sort examples by score
scored_examples = list(zip(example_scores, predictions, references, images))
scored_examples.sort(key=lambda x: x[0])

# Create output directories
output_dir = '/home/ldomene/CAP-GIA/blip/generation_examples'
os.makedirs(output_dir, exist_ok=True)
worst_dir = os.path.join(output_dir, 'worst_5')
best_dir = os.path.join(output_dir, 'best_5')
random_dir = os.path.join(output_dir, 'random_5')
os.makedirs(worst_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)
os.makedirs(random_dir, exist_ok=True)

# Save worst 5 examples
for i, (score, pred, ref, img) in enumerate(scored_examples[:5]):
    save_example(img, pred, ref, score, i, worst_dir)

# Save best 5 examples
for i, (score, pred, ref, img) in enumerate(scored_examples[-5:]):
    save_example(img, pred, ref, score, i, best_dir)

# Save 5 random examples
random_examples = random.sample(scored_examples, 5)
for i, (score, pred, ref, img) in enumerate(random_examples):
    save_example(img, pred, ref, score, i, random_dir)

print(f"Examples saved in {output_dir}")
