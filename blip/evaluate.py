from transformers import AutoModelForVision2Seq
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
model_finetuned = AutoModelForVision2Seq.from_pretrained("./training/caption")
model_finetuned.to(device)
