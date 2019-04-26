from modules.data_loader import COCODatasetProducer
import numpy as np
import os
import subprocess
from tqdm import tqdm

dtst_dir = '/playpen1/scribble/tianxian/dataset/MSCOCO2014/'


class CCGSuperTagging:
    def __init__(self):
        self.coco_producer = COCODatasetProducer(dtst_dir=dtst_dir, max_caption_length=20, vocabulary_size=5000)
        self.caption_path = os.path.join(dtst_dir,
                                         'all_captions_mcl_{}_vs_{}.txt'.format(self.coco_producer.max_caption_length,
                                                                                self.coco_producer.vocabulary_size))

    def extract_captions(self):
        captions, _, _ = self.coco_producer.build_train_imgid_imgfile_captions_file()
        captions = captions.tolist()
        print(np.shape(captions))
        print(captions[0])
        print(type(captions))
        with open(self.caption_path, 'w') as f:
            for cap in tqdm(captions):
                f.write(cap + '\n')

    def get_ccg_supertags(self):
        self.extract_captions()
        cmd = 'java -jar /playpen1/scribble/tianxian/EasySRL/easysrl.jar' \
              ' --outputFormat supertags --model /playpen1/scribble/tianxian/EasySRL/model/' \
              ' --f {}'.format(self.caption_path)
        print(cmd)
        cmd = cmd.split(' ')
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stderr.decode('utf-8'))

        caption_ccg_path = os.path.join(dtst_dir, 'all_captions_ccg_mcl_{}_vs_{}.txt'.format(
            self.coco_producer.max_caption_length, self.coco_producer.vocabulary_size))
        with open(caption_ccg_path, 'w') as f:
            f.write(result.stdout.decode('utf-8'))
