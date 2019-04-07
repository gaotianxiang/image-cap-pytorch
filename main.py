import argparse
from utils.utils import Params
import os
from director import Direcotr

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--model_dir', default='experiments/base_model')
args = parser.parse_args()

hps_path = os.path.join(args.model_dir, 'config.json')
if not os.path.exists(hps_path):
    raise FileNotFoundError('there is no config json file')
hps = Params(hps_path)
args.__dict__.update(hps.dict)


def train():
    director = Direcotr(gpu=args.gpu, pretrained_weight_dir=args.pretrained_weight_dir, dtst_dir=args.dtst_dir,
                        max_caption_length=args.max_caption_length, vocabulary_size=args.vocabulary_size,
                        hidden_size=args.hidden_size, teacher_forcing_ratio=args.teacher_forcing_ratio)
    director.train(epochs=args.epochs, lr=args.lr, log_every=args.log_every, batch_size=args.batch_size,
                   num_workers=args.num_workers)


def main():
    train()


if __name__ == '__main__':
    main()
