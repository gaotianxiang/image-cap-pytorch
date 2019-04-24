def get_language_decoder(hps):
    language_decoder = hps.language_decoder
    if language_decoder == 'vanilla':
        from .vanilla_decoder import LanguageDecoder
        return LanguageDecoder(hps)
    elif language_decoder == 'attention':
        from .attention_decoder import LanguageDecoder
        return LanguageDecoder(hps)
    else:
        raise ValueError('the type of language decoder is illegal')
