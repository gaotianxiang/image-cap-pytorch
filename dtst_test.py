from modules.data_loader import COCODatasetProducer
import torch.utils.data as data
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms
import torchvision


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

    dl = data.DataLoader(dtst, batch_size=1, shuffle=False, num_workers=16)

    with tqdm(total=len(dl)) as t:
        for _, _, _ in dl:
            t.update()


def img_test():
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open('./1.jpg').convert("RGB")
    img = trans(img)
    print(img.size())
    torchvision.utils.save_image(img, './resized.png')


if __name__ == '__main__':
    dtst_test()
    # img_test()
