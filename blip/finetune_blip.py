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

os.environ["CUDA_VISIBLE_DEVICES"]="1"


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
        encoding = self.processor(images = image , text= caption, padding= "max_length", return_tensors= "pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

transform = transforms.Compose([
	transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

"""
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						 std=[0.229, 0.224, 0.225])
])
"""


# Configuración del modelo BLIP
model_id = "model_resources"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForVision2Seq.from_pretrained(model_id)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Configuración PEFT
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "self.query",
        "self.key",
        "self.value",
        "output.dense",
        "self_attn.qkv",
        "self_attn.projection",
        "mlp.fc1",
        "mlp.fc2",
    ],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# Inicialización del dataset
data_path = "/home/ldomene/caption_data/receipes"
train_dataset = ReceipesDataset(data_path=data_path, transform_image=False, split="train")
val_dataset = ReceipesDataset(data_path=data_path, transform_image=False, split="val")

# DataLoader
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=5
)
val_dataloader = DataLoader(
    val_dataset, shuffle=False, batch_size=5
)

# Optimización
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

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
        "epochs": 8,
        "batch_size": 5,
        "learning_rate": 1e-4,
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
checkpoint_dir = "./checkpoints"
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
                pixel_values=val_pixel_values, max_new_tokens=64
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

    """# Save checkpoint
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
    print(f"Checkpoint saved: {checkpoint_path}")"""

    #Save model using model.save_pretrained
    path = f"/home/ldomene/CAP-GIA/blip/checkpoints/epoch_{epoch + 1}"
    model.save_pretrained(path)

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
os.makedirs("/home/ldomene/CAP-GIA/blip", exist_ok=True)
model.save_pretrained("/home/ldomene/CAP-GIA/blip/model")

artifact = wandb.Artifact("blip-finetuned-model", type="model")
artifact.add_dir("/home/ldomene/CAP-GIA/blip/model")
wandb.log_artifact(artifact)

wandb.finish()
