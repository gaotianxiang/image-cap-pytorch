def get_img_cap(hps):
    img_cap_type = hps.img_cap
    if img_cap_type == 'vanilla':
        from .img_cap import ImageCaptioning
        return ImageCaptioning(hps)
    elif img_cap_type == 'attention':
        from .attn_img_cap import ImageCaptioning
        return ImageCaptioning(hps)
    else:
        raise ValueError('the image captioning type is illegal')
