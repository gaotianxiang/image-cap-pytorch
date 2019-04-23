def get_img_encoder(hps):
    img_encoder = hps.img_encoder
    if img_encoder == 'vanilla':
        from .vanilla_encoder import CNN
        return CNN(hps)
    elif img_encoder == 'attention':
        from .attention_encoder import CNN
        return CNN(hps)
    else:
        raise ValueError('the type of image encoder is illegal')
