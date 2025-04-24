
import model.configs._configs as model_config


def got(type_str):
    if type_str == 'dict':
        return dict

    return getattr(model_config, type_str.split('.')[-1])
