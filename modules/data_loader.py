import torch.utils.data as data
import torch
import os
from .vocabulary import Vocabulary
from utils.coco.coco import COCO
import pandas as pd
from tqdm import tqdm
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


class SpecialTokens:
    PAD_token = 0
    SOS_token = 1
    EOS_toke = 2


class COCODataset(data.Dataset):
    def __init__(self, dtst_dir, max_caption_length, vocabulary_size, test_dir=None, split='train'):
        super().__init__()
        assert split in ['train', 'test', 'eval'], 'parameter ``split`` has to be one of train, eval, and val'
        self.dtst_dir = dtst_dir
        self.split = split
        self.test_dir = test_dir
        self.max_caption_length = max_caption_length
        self.train_caption_file_path = os.path.join(dtst_dir, 'annotations', 'captions_train2014.json')
        self.test_caption_file_path = os.path.join(dtst_dir, 'annotations', 'captions_val2014.json')
        self.vocabulary_file = os.path.join(dtst_dir, 'annotations', 'vocabulary.csv')
        self.train_anns_preprocess_file = os.path.join(dtst_dir, 'annotations', 'anns_preprocess.csv')
        self.train_anns_data_file = os.path.join(dtst_dir, 'annotations', 'train_data.npy')
        self.vocabulary_size = vocabulary_size
        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if split == 'train':
            self.img_fils, self.word_idxs, self.masks = self.prepare_train_data()
        elif split == 'eval':
            self.img_files = self.prepare_eval_data()
        elif split == 'test':
            if self.test_dir is None:
                raise Exception('no test dir as input')
            self.img_files = self.prepare_test_data()
        else:
            raise Exception('split must be one of train, eval, and test')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_fils[idx])
        if self.split == 'train':
            img = self.train_transform(img)
            return img, self.word_idxs[idx], self.masks[idx]
        elif self.split == 'eval':
            img = self.eval_transform(img)
            return img

    def build_vocabulary(self):
        coco = COCO(self.train_caption_file_path)
        coco.filter_by_cap_len(self.max_caption_length)

        vocabulary = Vocabulary(self.vocabulary_size)
        vocabulary.build(coco.all_captions())
        vocabulary.save(self.vocabulary_file)
        return vocabulary

    def prepare_train_data(self):
        coco_train = COCO(self.train_caption_file_path)
        coco_train.filter_by_cap_len(self.max_caption_length)

        print('building the vocabulary...')
        vocabulary = Vocabulary(self.vocabulary_size)
        if not os.path.exists(self.vocabulary_file):
            vocabulary.build(coco_train.all_captions())
            vocabulary.save(self.vocabulary_file)
        else:
            vocabulary.load(self.vocabulary_file)
        print('vocabulary built.')
        print('number of words = {}'.format(vocabulary.size))

        coco_train.filter_by_words(set(vocabulary.words))

        print('processing the captions...')
        if not os.path.exists(self.train_anns_preprocess_file):
            captions = [coco_train.anns[ann_id]['caption'] for ann_id in coco_train.anns]
            img_ids = [coco_train.anns[ann_id]['image_id'] for ann_id in coco_train.anns]
            img_files = [os.path.join(self.dtst_dir, 'train2014', coco_train.imgs[image_id]['file_name'])
                         for image_id in img_ids]
            annotations = pd.DataFrame({'image_id': img_ids,
                                        'image_file': img_files,
                                        'caption': captions})
            annotations.to_csv(self.train_anns_preprocess_file)
        else:
            annotations = pd.read_csv(self.train_anns_preprocess_file)
            captions = annotations['caption'].values
            img_ids = annotations['image_id'].values
            img_files = annotations['image_file'].values

        if not os.path.exists(self.train_anns_data_file):
            word_idxs = []
            masks = []
            for caption in tqdm(captions):
                current_word_idxs_ = vocabulary.process_sentence(caption)
                # Tokenize a sentence, and translate each token into its index in the vocabulary.
                current_num_words = len(current_word_idxs_)
                current_word_idxs = torch.zeros(self.max_caption_length, dtype=torch.long)
                current_masks = torch.zeros(self.max_caption_length)
                current_word_idxs[:current_num_words] = torch.tensor(current_word_idxs_)
                # current_word_idxs 共有 config.max_caption_length 长，其中前current_num_words位为对应句子的分词之后词语在vocabulary中的index
                current_masks[:current_num_words] = 1.0
                word_idxs.append(current_word_idxs)
                masks.append(current_masks)
            word_idxs = torch.cat(word_idxs)
            masks = torch.cat(masks)
            data = {'word_idxs': word_idxs, 'masks': masks}
            torch.save(data, self.train_anns_data_file)
        else:
            data = torch.load(self.train_anns_data_file)
            word_idxs = data['word_idxs']
            masks = data['masks']
        print('captions processed')
        print('number of captions = {}'.format(len(captions)))
        return img_files, word_idxs, masks

    def prepare_eval_data(self):
        coco_test = COCO(self.test_caption_file_path)
        img_ids = list(coco_test.imgs.keys())
        img_files = [os.path.join(self.dtst_dir, 'val2014', coco_test.imgs[image_id]['file_name']) for image_id in
                     img_ids]

        print('building the vocabulary')
        if os.path.exists(self.vocabulary_file):
            vocabulary = Vocabulary(self.vocabulary_size, self.vocabulary_file)
        else:
            vocabulary = self.build_vocabulary()
        print('vocabulary built.')
        print('number of words = {}'.format(vocabulary.size))
        return img_files

    def prepare_test_data(self):
        test_dir = self.test_dir
        files = os.listdir(test_dir)
        img_files = [os.path.join(test_dir, f) for f in files if
                     f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
        img_ids = list(range(len(img_files)))
        print("Building the vocabulary...")
        if os.path.exists(self.vocabulary_file):
            vocabulary = Vocabulary(self.vocabulary_size,
                                    self.vocabulary_file)
        else:
            vocabulary = self.build_vocabulary()
        print("Vocabulary built.")
        print("Number of words = {}".format(vocabulary.size))
        return img_files


def dtst_test():
    # train = COCODataset(dtst_dir='/playpen1/scribble/tianxian/dataset/MSCOCO2014/', max_caption_length=20,
    #                     vocabulary_size=5000, split='train')
    eval = COCODataset(dtst_dir='/playpen1/scribble/tianxian/dataset/MSCOCO2014/', max_caption_length=20,
                       vocabulary_size=5000, split='eval')


if __name__ == '__main__':
    dtst_test()
