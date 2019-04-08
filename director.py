import torch
from modules import ImageCaptioning, COCODatasetProducer, SparseCrossEntropy
import os
from tqdm import tqdm, trange
from utils.utils import RunningAverage
import torch.optim as optim
import torch.utils.data as data


class Direcotr:
    def __init__(self, gpu, pretrained_weight_dir, dtst_dir, max_caption_length, vocabulary_size,
                 hidden_size, teacher_forcing_ratio):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        os.environ['TORCH_MODEL_ZOO'] = pretrained_weight_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_producer = COCODatasetProducer(dtst_dir, max_caption_length, vocabulary_size)
        self.train_dtst = dataset_producer.prepare_train_data()
        self.eval_dtst = dataset_producer.prepare_eval_data()
        self.net = ImageCaptioning(device=self.device, hidden_size=hidden_size, max_length=max_caption_length,
                                   teacher_forcing_ratio=teacher_forcing_ratio, vocabulary_size=vocabulary_size,
                                   cnn_load_pretrained=True).to(self.device)
        self.loss_function = SparseCrossEntropy()

    def train(self, epochs, lr, log_every, batch_size, num_workers):
        ravg = RunningAverage()
        optimizer = optim.Adam(self.net.language_decoder.parameters(), lr=lr)
        dl = data.DataLoader(self.train_dtst, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             drop_last=True)
        self.net.language_decoder.train()
        self.net.image_encoder.eval()
        best_loss = 1e3

        for epoch in trange(epochs, desc='epochs'):
            ravg.reset()
            ite = 0
            with tqdm(total=len(self.train_dtst)) as progress_bar:
                for imgs, word_idxs, masks in dl:
                    optimizer.zero_grad()
                    imgs, word_idxs, masks = imgs.to(self.device), word_idxs.to(self.device), masks.to(self.device)
                    generated = self.net(imgs, word_idxs)
                    loss = self.loss_function(word_idxs, generated, masks)
                    loss.backward()
                    optimizer.step()
                    ite += 1
                    ravg.update(loss.item())
                    if ite % log_every == 0:
                        tqdm.write('epoch {} ite {} loss_average {:.5f}'.format(epoch, ite, ravg()))
                    if ite % 1000 == 0:
                        if ravg() < best_loss:
                            best_loss = ravg()
                            state_dict = {
                                'net': self.net.state_dict(),
                                'epoch': epoch,
                                'loss': ravg()
                            }
                            torch.save(state_dict, os.path.join('experiments/base_model', 'ckpts', 'best.pth.tar'))

                    progress_bar.set_postfix(loss_avg=ravg())
                    progress_bar.update(batch_size)

