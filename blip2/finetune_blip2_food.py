import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import wandb
import evaluate  
from typing import Literal

from transformers import Blip2Processor

MAX_LENGTH = 77

os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"


class Food500CapDataset(Dataset):
	def __init__(
			self,
			data_path: str,
			transform_image: bool = False,
			split: Literal["train", "val", "test"] = "train",
			split_size: list = [0.8, 0.2], # [train, val]
			data_size: int=1.0,
			return_img_path: bool = False,
			processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
	):
		super(Food500CapDataset, self).__init__()
		self.img_path = os.path.join(data_path, 'images')
		self.train_cap_path = os.path.join(data_path, "finetune_data.json")
		self.test_cap_path = os.path.join(data_path, "evaluation_data.json")
		train_cap_data = pd.read_json(self.train_cap_path)
		test_cap_data = pd.read_json(self.test_cap_path)
		self.transform_image = transform_image
		self.split = split
		self.return_img_path = return_img_path
		self.processor = processor

		# Clean data
		train_cap_data = train_cap_data.dropna(subset=["caption"])
		train_cap_data = train_cap_data[train_cap_data["caption"].apply(lambda x: len(x.split()) > 0)]
		test_cap_data = test_cap_data.dropna(subset=["caption"])
		test_cap_data = test_cap_data[test_cap_data["caption"].apply(lambda x: len(x.split()) > 0)]

		# Keep only data whose images exist
		train_cap_data = train_cap_data[train_cap_data["filename"].apply(lambda x: os.path.exists(os.path.join(self.img_path, x)))]
		test_cap_data = test_cap_data[test_cap_data["filename"].apply(lambda x: os.path.exists(os.path.join(self.img_path, x)))]

		# Split data
		total_size = len(train_cap_data)
		train_end = int(split_size[0] * total_size)

		if split == "train":
			self.cap_data = train_cap_data[:train_end]
		elif split == "val":
			self.cap_data = train_cap_data[train_end:]
		elif split == "test":
			self.cap_data = test_cap_data

		self.cap_data = self.cap_data.sample(frac=data_size, random_state=42)

	def __len__(self):
		return len(self.cap_data)
	
	def __getitem__(self, idx):
		img_name = os.path.join(self.img_path, self.cap_data.iloc[idx, 1])
		image = Image.open(img_name).convert('RGB')
		if self.transform_image:
			image = transform(image)
		caption = self.cap_data.iloc[idx, 2]
		encoding = self.processor(images = image , text= caption, padding= "max_length", return_tensors= "pt")
		encoding = {k:v.squeeze() for k,v in encoding.items()}
		return encoding

transform = transforms.Compose([
	transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir="/data3fast/users/elopez/models")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", quantization_config=quant_config, cache_dir="/data3fast/users/elopez/models")

# Configuración del modelo BLIP

device = "cuda" if torch.cuda.is_available() else "cpu"


from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# Inicialización del dataset
data_path = "/data3fast/users/elopez/food500cap/ISIA_Food500"
train_dataset = Food500CapDataset(data_path=data_path, transform_image=False, split="train", data_size=0.1)
val_dataset = Food500CapDataset(data_path=data_path, transform_image=False, split="val", data_size=0.1)

# DataLoader
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=10, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, shuffle=False, batch_size=10, collate_fn=collate_fn
)

# Optimización
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Inicialización de WandB
print("Wandb Iinitialization")
from dotenv import load_dotenv
# Load config
load_dotenv()
wandb_key = os.getenv("WANDB_KEY")
wandb.login(key=wandb_key)

wandb.init(
    project="CAP-GIA",
    config={
        "epochs": 10,
        "batch_size": 15,
        "learning_rate": 5e-4,
    },
)

config = wandb.config

# Métricas
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

# Entrenamiento
print("Start Training")

# Directory to save checkpoints
checkpoint_dir = "/data3fast/users/elopez/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print("Start Training")
for epoch in range(config.epochs):
    print(f"Epoch {epoch + 1}")
    total_loss = 0

    for idx, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})

    # Evaluation
    model.eval()
    predictions = []
    references = []
    images = []

    with torch.no_grad():
        for val_batch in val_dataloader:
            val_pixel_values = val_batch["pixel_values"].to(device)
            val_input_ids = val_batch["input_ids"].to(device)

            generated_output = model.generate(
                pixel_values=val_pixel_values, max_new_tokens=35
            )
            predictions.extend(
                processor.batch_decode(generated_output, skip_special_tokens=True)
            )
            references.extend(
                processor.batch_decode(val_input_ids, skip_special_tokens=True)
            )
            images.extend(val_pixel_values.cpu())  # Save images for visualization

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

    avg_train_loss = total_loss / len(train_dataloader)
    print(
        f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, BLEU-1 = {res_bleu_1['bleu']:.4f}, BLEU-2 = {res_bleu_2['bleu']:.4f}, METEOR = {res_meteor['meteor']:.4f}, ROUGE-L = {res_rouge['rougeL']:.4f}"
    )

    wandb.log(
        {
            "epoch_train_loss": avg_train_loss,
            "BLEU-1": res_bleu_1["bleu"],
            "BLEU-2": res_bleu_2["bleu"],
            "ROUGE-L": res_rouge["rougeL"],
            "METEOR": res_meteor["meteor"],
        }
    )

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_train_loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved: {checkpoint_path}")

    # Log example predictions in wandb every two epochs
    if (epoch + 1) % 2 == 0:
        example_logs = []
        num_examples = min(5, len(images))  # Number of examples to log
        for i in range(num_examples):
            # Prepare the image and captions
            image = images[i].permute(1, 2, 0).numpy()  # Convert to HWC format
            prediction = predictions[i]
            ground_truth = references[i]
            example_logs.append(
                wandb.Image(
                    image,
                    caption=f"Prediction: {prediction}\nGround Truth: {ground_truth}",
                )
            )
        
        wandb.log({f"Examples (Epoch {epoch + 1})": example_logs})

    model.train()
print("Finish Training")


# Guardar modelo
os.makedirs("/home/ldomene/CAP-GIA/blip2", exist_ok=True)
model.save_pretrained("/home/ldomene/CAP-GIA/blip2/last_model_saved")

artifact = wandb.Artifact("blip-finetuned-model", type="model")
artifact.add_dir("/home/ldomene/CAP-GIA/blip/model")
wandb.log_artifact(artifact)

wandb.finish()
