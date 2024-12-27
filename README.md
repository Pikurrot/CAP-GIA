# CAP-GIA
Image Captioning project for the Vision and Learning Subject of our AI Degree.

In this project, we implement a model that generates captions for images of food. The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images), it's a recipe dataset, where the Title (the caption we aim to generate) is the name of the recipe. It is not a trivial task, since many times the title is the technical name of the recipe, rather than a description of the food present in the image.

![food_image](github_images/github-image.jpg)

## Repository Structure
This repository contains the code used throughout the project, from the **first baseline models** designed to the final program that uses **the best model we finetunned** to generate captions for images. We also provide the **report** of the project, highlighting the process we followed and the results, and tje slides of the presentation:
- `process2job.sh` and `process2job.bat`: Scripts to run the training process.
- `config/train.yml`: The configuratio file used to train the models.
- `train.py`: Script that trains the model specified in `config/train.yml`.
- `src/`: Contains the source code of the models developed throughout the project.
- `blip/` and `blip2`: Contains the code of the BLIP and BLIP-2 models, including finetunning, evaluation and inference scripts.
- `Image_Captioning_Report.pdf`: The report of the project.
- `Image_Captioning_Slides.pdf`: The slides of the presentation.

## Usage
Setup your training configuration in `config/train.yml`. Then run the following command to start the training and evaluation process:

- Linux / MaxOS
```
bash process2job.sh
```
- Windows
```
process2job.bat
```

**Args (Optional):**
- `--gpu`: The GPU to use. Default is -1 (use all available GPUs). E.g. `--gpu 4`.
- `--data_size`: The fraction of the dataset to train with. Default is 1.0 (use the whole dataset). E.g. `--data_size 0.25`.

Also, update the paths **OUT_DIR** and **DATA_DIR** in the process2job script to your own paths.  
If you train in a cluster with SLURM, add these lines right after `#!/bin/bash` in the `process2job.sh` script:
```
#SBATCH -n 4 # Request 4 CPU' s cores . Maximum 10 CPU ’ s cores.
#SBATCH -N 1 # Ensure that all cores are on one machine.
#SBATCH -D /fhome/vlia01/CAP-GIA # Working directory. Change to your user homer>
#SBATCH -t 4-00:05 # Runtime in D - HH : MM
#SBATCH -p tfg # Partition to submit to.
#SBATCH --mem 12288 # Request 12 GB of RAM memory. Maximum 60 GB.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written.
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written.
#SBATCH --gres gpu:1 # Request 1 GPU. Maximum 8 GPUs.

sleep 3
```

**! ! !** Additionally, if you have a [W&B](https://wandb.ai/site) account, you can track the training and evaluation of the model. Just set `log_wandb: True` in the `config/train.yml` file, and create a `.env` file in the root directory with the line `WANDB_KEY=<your_key>`.


## Finetuned Models
Find our finetuned models on Hugging Face:  
- [BLIP](https://huggingface.co/luisdomene4/BLIP-Finetune-Recipes)
- [BLIP2](https://huggingface.co/luisdomene4/BLIP2-Finetune-Recipes)
