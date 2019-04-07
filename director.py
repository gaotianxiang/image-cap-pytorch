import torch
from modules import ImageCaptioning, COCODatasetProducer, SparseCrossEntropy
import os


class Direcotr:
    def __init__(self, gpu, pretrained_weight_dir, dtst_dir, max_caption_length, vocabulary_size,
                 hidden_size, teacher_forcing_ratio):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        os.environ['TORCH_MODEL_ZOO'] = pretrained_weight_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_producer = COCODatasetProducer(dtst_dir, max_caption_length, vocabulary_size)
        self.train_dl = dataset_producer.prepare_train_data()
        self.eval_dl = dataset_producer.prepare_eval_data()
        self.net = ImageCaptioning(device=self.device, hidden_size=hidden_size, max_length=max_caption_length,
                                   teacher_forcing_ratio=teacher_forcing_ratio, vocabulary_size=vocabulary_size,
                                   cnn_load_pretrained=True)
        self.loss_function = SparseCrossEntropy()

    def train(self, epochs, lr, log_every):
        raise NotImplementedError


