import torch.utils.data as data
import torch
import os
from .vocabulary import Vocabulary
from utils.coco.coco import COCO
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import transforms
from PIL import Image


class SpecialTokens:
    PAD_token = 0
    SOS_token = 1
    EOS_toke = 2


class COCODatasetProducer:
    def __init__(self, dtst_dir, max_caption_length, vocabulary_size):
        super().__init__()
        self.dtst_dir = dtst_dir
        self.max_caption_length = max_caption_length

        self.train_caption_file_path = os.path.join(dtst_dir, 'annotations', 'captions_train2014.json')
        self.test_caption_file_path = os.path.join(dtst_dir, 'annotations', 'captions_val2014.json')

        self.vocabulary_size = vocabulary_size

    def is_vocabulary_file_exists(self):
        file_name = 'vocabulary_size_{}_length_{}.csv'.format(self.vocabulary_size, self.max_caption_length)
        file_path = os.path.join(self.dtst_dir, 'annotations', file_name)
        return os.path.exists(file_path), file_path

    def build_vocabulary(self):
        file_exists, file_path = self.is_vocabulary_file_exists()
        if file_exists:
            print('vocabulary has been built, now loading it...')
            vocabulary = Vocabulary(self.vocabulary_size)
            vocabulary.load(file_path)
            return vocabulary
        else:
            coco = COCO(self.train_caption_file_path)
            coco.filter_by_cap_len(self.max_caption_length)

            print('building the vocabulary...')
            vocabulary = Vocabulary(self.vocabulary_size)
            vocabulary.build(coco.all_captions())
            vocabulary.save(file_path)
            print('vocabulary built.')
            return vocabulary

    def is_train_imgid_imgfile_captions_file_exists(self):
        file_name = 'anns_detail_size_{}_length_{}.csv'.format(self.vocabulary_size, self.max_caption_length)
        file_path = os.path.join(self.dtst_dir, 'annotations', file_name)
        return os.path.exists(file_path), file_path

    def build_train_imgid_imgfile_captions_file(self):
        file_exists, file_path = self.is_train_imgid_imgfile_captions_file_exists()

        if file_exists:
            annotations = pd.read_csv(file_path)
            captions = annotations['caption'].values
            img_ids = annotations['image_id'].values
            img_files = annotations['image_file'].values
            return captions, img_ids, img_files
        else:
            coco_train = COCO(self.train_caption_file_path)
            coco_train.filter_by_cap_len(self.max_caption_length)
            print('loading vocabulary...')
            vocabulary = self.build_vocabulary()
            coco_train.filter_by_words(set(vocabulary.words))

            print('processing the captions...')
            captions = [coco_train.anns[ann_id]['caption'] for ann_id in coco_train.anns]
            img_ids = [coco_train.anns[ann_id]['image_id'] for ann_id in coco_train.anns]
            img_files = [os.path.join(self.dtst_dir, 'train2014', coco_train.imgs[image_id]['file_name'])
                         for image_id in img_ids]
            annotations = pd.DataFrame({'image_id': img_ids,
                                        'image_file': img_files,
                                        'caption': captions})
            annotations.to_csv(file_path)
            print('annotations img_ids, img_files, and captions file built.')
            return captions, img_ids, img_files

    def is_train_anns_masks_file_exists(self):
        file_name = 'train_anns_masks_size_{}_length_{}.pth.tar'.format(self.vocabulary_size, self.max_caption_length)
        file_path = os.path.join(self.dtst_dir, 'annotations', file_name)
        return os.path.exists(file_path), file_path

    def build_train_anns_masks_file(self, captions, vocabulary):
        file_exists, file_path = self.is_train_anns_masks_file_exists()
        if file_exists:
            data = torch.load(file_path)
            word_idxs = data['word_idxs']
            masks = data['masks']
        else:
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
            word_idxs = torch.stack(word_idxs)
            masks = torch.stack(masks)
            data = {'word_idxs': word_idxs, 'masks': masks}
            torch.save(data, file_path)
        return word_idxs, masks

    def prepare_train_data(self):
        vocabulary = self.build_vocabulary()
        captions, img_ids, img_files = self.build_train_imgid_imgfile_captions_file()
        word_idxs, masks = self.build_train_anns_masks_file(captions, vocabulary)
        print('captions processed')
        print('number of captions = {}'.format(len(captions)))
        coco_train_dtst = COCOTrain(img_files, word_idxs, masks)
        return coco_train_dtst

    def prepare_eval_data(self):
        coco_test = COCO(self.test_caption_file_path)
        img_ids = list(coco_test.imgs.keys())
        img_files = [os.path.join(self.dtst_dir, 'val2014', coco_test.imgs[image_id]['file_name']) for image_id in
                     img_ids]

        print('building the vocabulary')
        vocabulary = self.build_vocabulary()
        print('vocabulary built.')
        print('number of words = {}'.format(vocabulary.size))

        coco_eval_dtst = COCOEval(coco_test, img_files, vocabulary)
        return coco_eval_dtst

    def prepare_test_data(self, test_dir):
        files = os.listdir(test_dir)
        img_files = [os.path.join(test_dir, f) for f in files if
                     f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
        img_ids = list(range(len(img_files)))
        print("Building the vocabulary...")
        vocabulary = self.build_vocabulary()
        print("Vocabulary built.")
        print("Number of words = {}".format(vocabulary.size))

        test_dtst = Test(img_files, vocabulary)
        return test_dtst


class COCOTrain(data.Dataset):
    def __init__(self, img_files, word_idxs, masks):
        super(COCOTrain, self).__init__()
        self.img_files = img_files
        self.word_idxs = word_idxs
        self.masks = masks
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        return self.transform(img), self.word_idxs[idx], self.masks[idx]


class COCOEval(data.Dataset):
    def __init__(self, eval_coco, img_files, vocabulary):
        super(COCOEval, self).__init__()
        self.eval_coco = eval_coco
        self.img_files = img_files
        self.vocabulary = vocabulary
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        return self.transform(img)


class Test(data.Dataset):
    def __init__(self, img_files, vocabulary):
        super(Test, self).__init__()
        self.img_files = img_files
        self.vocabulary = vocabulary
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        return self.transform(img)
