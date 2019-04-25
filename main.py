import argparse
from utils.utils import Params
import os
from director import Director


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--model_dir', '--md', default='experiments/base_model', type=str)
    parser.add_argument('--resume', '--r', action='store_true')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--test_dir', default='test', type=str)
    parser.add_argument('--beam_size', '--bs', default=3, type=int)
    args = parser.parse_args()

    hps_path = os.path.join(args.model_dir, 'config.json')
    if not os.path.exists(hps_path):
        raise FileNotFoundError('there is no config json file')
    hps = Params(hps_path)
    hps.dict.update(args.__dict__)
    # args.__dict__.update(hps.dict)
    if hps.mode != 'train':
        hps.batch_size = 1
    return hps


def main():
    args = get_parameters()
    director = Director(args)
    if args.mode == 'train':
        director.train()
    elif args.mode == 'eval':
        director.eval()
    elif args.mode == 'test':
        director.test()


if __name__ == '__main__':
    main()
