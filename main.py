import os
import torch
import yaml
from easydict import EasyDict as edict
from datetime import datetime

import ctools
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.base_network import EfficientNet, ResNet
from trainer import BYOLTrainer
from data.reader import loader

# print(torch.__version__)
torch.manual_seed(0)


def main():
    config = edict(yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader))

    data = config.data
    save = config.save
    batch_size = config.trainer.batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])
    transform = MultiViewDataInjector([data_transform, data_transform])

    if data.isFolder:
        data, _ = ctools.readfolder(data)

    train_loader = loader(data, batch_size, transform, shuffle=True, num_workers=2)

    # online network
    if "resnet" in config['network']['name']:
        online_network = ResNet(**config['network']).to(device)
    elif "efficientnet" in config['network']['name']:
        online_network = EfficientNet(**config['network']).to(device)
    else:
        raise ValueError(f"Model {config['network']['name']} not available.")
    pretrained_path = config['network']['pretrain']

    # load pre-trained model if defined
    if pretrained_path:
        try:
            # load pre-trained parameters
            load_params = torch.load(pretrained_path, map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    if "resnet" in config['network']['name']:
        target_network = ResNet(**config['network']).to(device)
    elif "efficientnet" in config['network']['name']:
        target_network = EfficientNet(**config['network']).to(device)
    else:
        raise ValueError(f"Model {config['network']['name']} not available.")

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    
    log_dir = os.path.join(save.metapath, data.name, config['network']['name'])
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          log_dir=log_dir,
                          **config['trainer'])

    trainer.train(train_loader)


if __name__ == '__main__':
    main()
