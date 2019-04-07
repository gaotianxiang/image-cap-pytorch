from modules.data_loader import COCODatasetProducer


def dtst_test():
    train = COCODatasetProducer(dtst_dir='/playpen1/scribble/tianxian/dataset/MSCOCO2014/', max_caption_length=20,
                                vocabulary_size=5000)
    dtst = train.prepare_train_data()
    eval = train.prepare_eval_data()
    test = train.prepare_test_data(test_dir='./test')
    print(dtst[0])
    a, b, c = dtst[0]
    print(len(dtst))
    print(a.size())
    print(b.size())
    print(c.size())
    print(len(eval))
    print(eval[0])
    print(eval[0].size())

    print(len(test))
    print(test[0].size())


if __name__ == '__main__':
    dtst_test()
