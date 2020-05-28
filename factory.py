import torch


def load_params(net, path):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    w_dict = torch.load(path)
    for k, v in w_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)


def load_data_set_from_factory(configs, phase):
    if configs['db']['name'] == 'mvtec':
        from db import MVTEC, MVTEC_pre
        if phase == 'train':
            set_name = configs['db']['train_split']
        elif phase == 'test':
            set_name = configs['db']['val_split']
        else:
            raise Exception("Invalid phase name")
        set = MVTEC(root=configs['db']['data_dir'], set=set_name, preproc=MVTEC_pre(resize=None))

    else:
        raise Exception("Invalid set name")

    return set


def load_training_net_from_factory(configs):
    if configs['model']['name'] == 'SSIM_AE':
        from model.networks import AE_basic
        net = AE_basic(img_channel=configs['model']['img_channel'])
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))

        return net, optimizer

    else:
        raise Exception("Invalid model name")


def load_loss_from_factory(configs):
    if configs['op']['loss'] == 'SSIM_loss':
        from model.loss import SSIM_loss
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=configs['model']['img_channel'])

        return loss

    else:
        raise Exception('Wrong loss name')


def load_training_model_from_factory(configs, ngpu):
    if configs['model']['type'] == 'AutoEncoder':
        net, optimizer = load_training_net_from_factory(configs)
        loss = load_loss_from_factory(configs)
        from model import AE_trainer as Trainer
        trainer = Trainer(net, loss, configs['op']['loss'], optimizer, ngpu)

    else:
        raise Exception("Wrong model type!")

    return trainer


def load_test_model_from_factory(configs):
    if configs['model']['name'] == 'SSIM_Net':
        from model.networks import SSIM_Net
        net = SSIM_Net(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])

    else:
        raise Exception("Invalid model name")

    return net
