import torch
from modules import ImageCaptioning, COCODatasetProducer, SparseCrossEntropy
import os
from tqdm import tqdm, trange
from utils.utils import RunningAverage
import torch.optim as optim
import torch.utils.data as data


class Direcotr:
    def __init__(self, gpu, pretrained_weight_dir, dtst_dir, max_caption_length, vocabulary_size,
                 hidden_size, teacher_forcing_ratio, model_dir):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        os.environ['TORCH_MODEL_ZOO'] = pretrained_weight_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.ckpts_dir = os.path.join(model_dir, 'ckpts')
        self.start_epoch = 0

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

        for epoch in trange(self.start_epoch, self.start_epoch + epochs, desc='epochs'):
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

    def load_ckpts(self):
        ckpts = os.path.join(self.ckpts_dir, 'best.pth.tar')
        if not os.path.exists(ckpts):
            raise FileNotFoundError('there is no ckpts file in the directory {}'.format(self.ckpts_dir))
        state_dict = torch.load(ckpts)
        self.net.load_state_dict(state_dict['net'])
        self.start_epoch = state_dict['epoch'] + 1
        print('has already trained for {} epochs and the loss is {:.4f}'.format(state_dict['epoch'],
                                                                                state_dict['loss']))
        return state_dict['epoch'], state_dict['loss']
