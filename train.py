import argparse
from src.ResnetCaption import ResnetLSTMCaption

def train(
		model_name: str,
		dataset_name: str,
		output_dir: str,
		data_dir: str
):
	if model_name.startswith("resnet"):
		model = ResnetLSTMCaption(
			resnet_name=model_name,
			output_dir=output_dir
		)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='resnet18')
	parser.add_argument('--dataset', type=str, default='flickr-8k')
	parser.add_argument('--out_dir', type=str, default='out')
	parser.add_argument("--data_dir", type=str, default="")
	args = parser.parse_args()
	
	assert args.model in ["resnet18", "resnet34", "resnet50"]
	assert args.dataset in ["flickr-8k", "flickr-30k"]

	train(
		model_name=args.model,
		dataset_name=args.dataset,
		output_dir=args.out_dir,
		data_dir=args.data_dir
	)
