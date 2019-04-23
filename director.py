import torch
from modules import ImageCaptioning, COCODatasetProducer, SparseCrossEntropy
import os
from tqdm import tqdm, trange
from utils.utils import RunningAverage
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils.utils import set_logger, log
import json
from utils.coco.pycocoevalcap.eval import COCOEvalCap


class Director:
    def __init__(self, hps):
        self.hps = hps
        os.environ['CUDA_VISIBLE_DEVICES'] = hps.gpu
        os.environ['TORCH_MODEL_ZOO'] = hps.pretrained_weight_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = hps.model_dir
        self.start_epoch = 0
        self.max_caption_length = hps.max_caption_length
        self.hidden_size = hps.hidden_size

        self.dataset_producer = COCODatasetProducer(hps.dtst_dir, hps.max_caption_length, hps.vocabulary_size)
        # self.train_dtst = self.dataset_producer.prepare_train_data()
        # self.eval_dtst = self.dataset_producer.prepare_eval_data()
        self.net = ImageCaptioning(device=self.device, hidden_size=hps.hidden_size, max_length=hps.max_caption_length,
                                   teacher_forcing_ratio=hps.teacher_forcing_ratio, vocabulary_size=hps.vocabulary_size,
                                   cnn_load_pretrained=True).to(self.device)
        self.loss_function = SparseCrossEntropy()

    def train(self):
        set_logger(os.path.join(self.log_dir, 'train.log'), terminal=False)
        epochs = self.hps.epochs
        lr = self.hps.lr
        log_every = self.hps.log_every
        save_every = self.hps.save_every
        batch_size = self.hps.batch_size
        num_workers = self.hps.num_workers
        train_dtst = self.dataset_producer.prepare_train_data()
        ravg = RunningAverage()
        optimizer = optim.Adam(params=[
            {'params': self.net.language_decoder.parameters()},
            {'params': self.net.image_encoder.fcn.parameters()}
        ], lr=lr)
        dl = data.DataLoader(train_dtst, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             drop_last=True)
        self.net.train()
        self.net.image_encoder.inception.eval()
        best_loss = 1e3
        global_step = 0

        if self.hps.resume:
            _, _, global_step = self.load_ckpts()

        for epoch in trange(self.start_epoch, self.start_epoch + epochs, desc='epochs'):
            ravg.reset()
            with tqdm(total=len(train_dtst)) as progress_bar:
                for imgs, word_idxs, masks in dl:
                    optimizer.zero_grad()
                    imgs, word_idxs, masks = imgs.to(self.device), word_idxs.to(self.device), masks.to(self.device)
                    generated = self.net(imgs, word_idxs)
                    loss = self.loss_function(word_idxs, generated, masks)
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    ravg.update(loss.item())
                    if global_step % log_every == 0:
                        log('epoch {} ite {} loss_average {:05.5f}'.format(epoch, global_step, ravg()))
                    if global_step % save_every == 0:
                        if ravg() < best_loss:
                            best_loss = ravg()
                            state_dict = {
                                'net': self.net.state_dict(),
                                'epoch': epoch,
                                'loss': ravg(),
                                'global_step': global_step
                            }
                            torch.save(state_dict, os.path.join(self.ckpts_dir, 'best.pth.tar'))

                    progress_bar.set_postfix(loss_avg='{:05.5f}'.format(ravg()))
                    progress_bar.update(batch_size)

    def eval(self):
        set_logger(os.path.join(self.log_dir, 'eval.log'), terminal=False)
        log('- evaluating the model on coco dataset')
        eval_dtst = self.dataset_producer.prepare_eval_data()
        eval_dl = data.DataLoader(eval_dtst, batch_size=1)
        eval_coco = eval_dtst.eval_coco
        vocabulary = eval_dtst.vocabulary
        results = []

        # _, _, global_step = self.load_ckpts()
        self.load_ckpts()
        caption_path = os.path.join(self.eval_dir, 'eval_caption.json')
        if os.path.exists(caption_path):
            log('- already generate captions')
            eval_result_coco = eval_coco.loadRes(caption_path)
            scorer = COCOEvalCap(eval_coco, eval_result_coco)
            scorer.evaluate()
            log("- Evaluation complete.")
            log('--------------------------------------')
            return
        self.net.eval()
        with torch.no_grad():
            with tqdm(total=len(eval_dl)) as progress_bar:
                for id, img in eval_dl:
                    img = img.to(self.device)
                    fvs = self.net.image_encoder(img)
                    sentence = []
                    decoder_input = torch.tensor([0] * 1, dtype=torch.long, device=self.device)
                    decoder_hidden = fvs.view(1, 1, self.hidden_size)

                    for _ in range(self.max_caption_length):
                        decoder_output, decoder_hidden = self.net.language_decoder(decoder_input, decoder_hidden)
                        topv, topi = decoder_output.topk(1)
                        if vocabulary.words[topi.item()] == '.':
                            break
                        else:
                            sentence.append(topi.item())
                            decoder_input = topi.detach()
                    caption = vocabulary.get_sentence(sentence)
                    results.append({'image_id': int(id),
                                    'caption': caption})
                    progress_bar.update()
        with open(caption_path, 'w') as f:
            json.dump(results, f)

        # # Evaluate these captions
        eval_result_coco = eval_coco.loadRes(caption_path)
        scorer = COCOEvalCap(eval_coco, eval_result_coco)
        scorer.evaluate()
        log("- Evaluation complete.")
        log('--------------------------------------')

    def test(self):
        test_dir = self.hps.test_dir
        test_dir = os.path.join(self.model_dir, test_dir)
        if not os.path.exists(test_dir):
            raise FileNotFoundError('there are no such directory {}'.format(test_dir))
        test_result_dir = os.path.join(self.model_dir, 'test_result')
        os.makedirs(test_result_dir, exist_ok=True)
        test_dtst = self.dataset_producer.prepare_test_data(test_dir=test_dir)
        test_dl = data.DataLoader(test_dtst, batch_size=1)
        vocabulary = test_dtst.vocabulary

        inv_trans = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.ToPILImage()
        ])

        self.load_ckpts()
        self.net.eval()
        with torch.no_grad():
            with tqdm(total=len(test_dl)) as progress_bar:
                for i, img in enumerate(test_dl):
                    img = img.to(self.device)
                    fvs = self.net.image_encoder(img)
                    sentence = []
                    decoder_input = torch.tensor([0] * 1, dtype=torch.long, device=self.device)
                    decoder_hidden = fvs.view(1, 1, self.hidden_size)

                    for _ in range(self.max_caption_length):
                        decoder_output, decoder_hidden = self.net.language_decoder(decoder_input, decoder_hidden)
                        topv, topi = decoder_output.topk(1)
                        if vocabulary.words[topi.item()] == '.':
                            break
                        else:
                            sentence.append(topi.item())
                            decoder_input = topi.detach()
                    caption = vocabulary.get_sentence(sentence)
                    img = img.to('cpu')
                    img = inv_trans(img[0])
                    plt.imshow(img)
                    plt.title(caption)
                    plt.axis('off')
                    plt.savefig(os.path.join(test_result_dir, '{}.png'.format(i)))
                    progress_bar.update()

    def load_ckpts(self):
        ckpts = os.path.join(self.ckpts_dir, 'best.pth.tar')
        if not os.path.exists(ckpts):
            raise FileNotFoundError('there is no ckpts file in the directory {}'.format(self.ckpts_dir))
        log('- load ckpts from {}'.format(ckpts))
        state_dict = torch.load(ckpts)
        self.net.load_state_dict(state_dict['net'])
        self.start_epoch = state_dict['epoch'] + 1
        log('- already trained for {} epochs and the loss is {:.4f}'.format(state_dict['epoch'],
                                                                            state_dict['loss']))
        return state_dict['epoch'], state_dict['loss']  # , state_dict['global_step']

    @property
    def log_dir(self):
        log_dir = os.path.join(self.model_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @property
    def ckpts_dir(self):
        ckpts_dir = os.path.join(self.model_dir, 'ckpts')
        os.makedirs(ckpts_dir, exist_ok=True)
        return ckpts_dir

    @property
    def eval_dir(self):
        eval_dir = os.path.join(self.model_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir
