import torch
from modules import get_img_cap, COCODatasetProducer, SparseCrossEntropy
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
        hps.device = self.device
        self.model_dir = hps.model_dir
        self.start_epoch = 0
        self.max_caption_length = hps.max_caption_length
        self.hidden_size = hps.hidden_size

        self.dataset_producer = COCODatasetProducer(hps.dtst_dir, hps.max_caption_length, hps.vocabulary_size)

        self.net = get_img_cap(hps).to(self.device)
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
        if self.hps.img_cap == 'vanilla':
            optimizer = optim.Adam(params=[
                {'params': self.net.language_decoder.parameters()},
                {'params': self.net.img_fvs_to_hs.parameters()}
            ], lr=lr)
        elif self.hps.img_cap == 'attention':
            optimizer = optim.Adam(params=self.net.language_decoder.parameters(), lr=lr)
        else:
            raise ValueError('the image captioning type is illegal')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, threshold=1e-2, patience=3,
                                                               verbose=True)
        dl = data.DataLoader(train_dtst, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             drop_last=True)
        self.net.train()
        self.net.image_encoder.eval()
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
                        ckpt_path = os.path.join(self.ckpts_dir, 'ckpt_{}.pth.tar'.format(global_step))
                        state_dict = {
                            'net': self.net.state_dict(),
                            'epoch': epoch,
                            'loss': ravg(),
                            'global_step': global_step
                        }
                        torch.save(state_dict, ckpt_path)
                        delete_ckpt_path = os.path.join(self.ckpts_dir, 'ckpt_{}.pth.tar'.format(
                            global_step - self.hps.num_ckpts_saved * save_every))
                        if os.path.exists(delete_ckpt_path):
                            os.remove(delete_ckpt_path)
                        scheduler.step(ravg())

                    progress_bar.set_postfix(loss_avg='{:05.5f}'.format(ravg()))
                    progress_bar.update(batch_size)
            if ravg() < best_loss:
                best_loss = ravg()
                state_dict = {
                    'net': self.net.state_dict(),
                    'epoch': epoch,
                    'loss': ravg(),
                    'global_step': global_step
                }
                torch.save(state_dict, os.path.join(self.ckpts_dir, 'best.pth.tar'))
            log('- epoch {} done loss {:05.5f}'.format(epoch, ravg()))
            # scheduler.step(ravg())

    def beam_search(self, img, vocabulary):
        fvs = self.net.image_encoder(img)
        decoder_hidden = self.net.img_fvs_to_hs(fvs)
        decoder_input = torch.tensor([0] * 1, dtype=torch.long, device=self.device)
        beam_size = self.hps.beam_size

        beam_partial_sentences = [[list(), decoder_input, decoder_hidden, 0, 0]] * beam_size
        for di in range(self.max_caption_length):
            if di == 0:
                decoder_output, decoder_hidden = self.net.language_decoder(decoder_input, decoder_hidden, fvs)
                topv, topi = decoder_output.squeeze().topk(beam_size)

                for k in range(beam_size):
                    beam_partial_sentences = [
                        [[topi[k].item()],
                         topi[k].detach(),
                         decoder_hidden,
                         topv[k].item(),
                         1 if vocabulary.words[topi[k].item()] == '.' else 0]
                        for k in [0, 1, 2]
                    ]
                # print()
            else:
                for i in range(beam_size):
                    if beam_partial_sentences[i][4] == 1:
                        beam_partial_sentences.append(beam_partial_sentences[i])
                        continue
                    decoder_output, decoder_hidden = self.net.language_decoder(
                        beam_partial_sentences[i][1],
                        beam_partial_sentences[i][2],
                        fvs
                    )
                    topv, topi = decoder_output.squeeze().topk(beam_size)
                    local_beam_partial_sentences = [
                        [beam_partial_sentences[i][0] + [topi[k].item()],
                         topi[k].detach(),
                         decoder_hidden,
                         beam_partial_sentences[i][3] + topv[k].item(),
                         1 if vocabulary.words[topi[k].item()] == '.' else 0]
                        for k in range(beam_size)
                    ]
                    beam_partial_sentences += local_beam_partial_sentences
                beam_partial_sentences = beam_partial_sentences[beam_size:]
                beam_partial_sentences = sorted(beam_partial_sentences, key=lambda x: x[3], reverse=True)[
                                         0:beam_size]
        return beam_partial_sentences[0][0]

    def greedy_search(self, img, vocabulary):
        fvs = self.net.image_encoder(img)
        decoder_hidden = self.net.img_fvs_to_hs(fvs)
        decoder_input = torch.tensor([0] * 1, dtype=torch.long, device=self.device)
        sentence = []
        for di in range(self.hps.max_caption_length):
            decoder_output, decoder_hidden = self.net.language_decoder(decoder_input, decoder_hidden, fvs)
            topv, topi = decoder_output.topk(1)
            if vocabulary.words[topi.item()] == '.':
                break
            else:
                sentence.append(topi.item())
                decoder_input = topi.detach()
        return sentence

    def eval(self):
        set_logger(os.path.join(self.log_dir, 'eval.log'), terminal=False)
        log('- evaluating the model on coco dataset')
        eval_dtst = self.dataset_producer.prepare_eval_data()
        eval_dl = data.DataLoader(eval_dtst, batch_size=1, num_workers=self.hps.num_workers)
        eval_coco = eval_dtst.eval_coco
        vocabulary = eval_dtst.vocabulary
        results = []
        beam_size = self.hps.beam_size
        _, _, global_step = self.load_ckpts()

        if beam_size != 0:
            log('- beam search, beam size is {}'.format(beam_size))
            caption_path = os.path.join(self.eval_dir,
                                        'eval_caption_globalstep_{}_beam_{}.json'.format(global_step, beam_size))
        else:
            log('- greedy search')
            caption_path = os.path.join(self.eval_dir, 'eval_caption_globalstep_{}.json'.format(global_step))
        log('- the caption will be stored in file {}'.format(caption_path))
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
                    if beam_size > 0:
                        sentence = self.beam_search(img, vocabulary)
                    else:
                        sentence = self.greedy_search(img, vocabulary)
                    # fvs = self.net.image_encoder(img)
                    # fvs = self.net.img_fvs_to_hs(fvs)
                    # sentence = []
                    # decoder_input = torch.tensor([0] * 1, dtype=torch.long, device=self.device)
                    # decoder_hidden = fvs.view(1, 1, self.hidden_size)

                    # for di in range(self.hps.max_length):
                    #     decoder_output, decoder_hidden = self.net.language_decoder(decoder_input, decoder_hidden)
                    #     topv, topi = decoder_output.topk(1)
                    #     if vocabulary.words[topi.item()] == '.':
                    #         break
                    #     else:
                    #         sentence.append(topi.item())
                    #         decoder_input = topi.detach()
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
        return state_dict['epoch'], state_dict['loss'], state_dict['global_step']

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
