# CAP-GIA
Image Captioning project for the Vision and Learning Subject of our AI Degree

## Usage
- Linux / MaxOS
```
bash process2job.sh --[train|test] --model <model_name> --dataset <dataset_name>
```
- Windows
```
process2job.bat --[train|test] --model <model_name> --dataset <dataset_name>
```

**Args:**
- `--train` or `--test`: Run the train or test script.
- `--model`: The name of the model to use. E.g. **resnet-18**.
- `--dataset`: The name of the dataset to use. E.g. **flickr-8k**.

Also, update the paths **OUT_DIR** and **DATA_DIR** in the process2job script to your own paths.
