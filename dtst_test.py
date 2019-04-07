from modules.data_loader import COCODataset


def dtst_test():
    # train = COCODataset(dtst_dir='/playpen1/scribble/tianxian/dataset/MSCOCO2014/', max_caption_length=20,
    #                     vocabulary_size=5000, split='train')
    eval = COCODataset(dtst_dir='/playpen1/scribble/tianxian/dataset/MSCOCO2014/', max_caption_length=20,
                       vocabulary_size=5000, split='eval')


if __name__ == '__main__':
    dtst_test()
