from modules.data_loader import COCODatasetProducer
import torch.utils.data as data
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt


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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inv_trans = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage()
    ])
    # inv_trans = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    img = Image.open('./4.jpg')
    img = trans(img)
    print(img.size())

    # torchvision.utils.save_image(img, './resized.png')
    img = inv_trans(img)
    # print(img)
    img = np.array(img)
    print(np.shape(img))
    plt.imshow(img)
    plt.title('test')
    plt.axis('off')
    plt.savefig('./test.png')
    # img[0] = img[0] * 0.229 + 0.485
    # img[1] = img[1] * 0.224 + 0.456
    # img[2] = img[2] * 0.225 + 0.406
    # torchvision.utils.save_image(img, './back.jpg')


if __name__ == '__main__':
    # dtst_test()
    img_test()
