import argparse
from pathlib import Path

import tensorflow as tf

tf.config.run_functions_eagerly(True)

from dataset import DataFactory
from models import MISLNet, MobileNet, EfficientNet, MobileNetContrastive, ResNetContrastive, EfficientNetB0Contrastive


def none_or_str(value):
    if value == 'None':
        return None
    return str(value)


def none_or_bool(value):
    if value == 'None':
        return None
    return bool(int(value))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the CNNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fc_layers', type=int, default=2, help='Number of fully-connected layers [default: 2]')
    parser.add_argument('--fc_size', type=int, default=1024, help='Number of neurons in Fully Connected layers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', type=str, help='Name for the model')
    parser.add_argument('--use_pretrained', type=none_or_bool, default=True,
                        help='Use pretrained weights from ImageNet')
    parser.add_argument('--model_path', type=str, help='Path to model to continue training (*.h5)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset to train the constrained_net')
    parser.add_argument('--height', type=int, default=480, help='Height of CNN input dimension [default: 480]')
    parser.add_argument('--width', type=int, default=800, help='Width of CNN input dimension [default: 800]')
    parser.add_argument('--frames_per_video', type=int, default=50,
                        help='to determine #frames to sample for creating multi-frame dataset for the ConvNet')
    parser.add_argument('--gpu_id', type=int, default=None, help='Choose the available GPU devices')
    parser.add_argument('--category', type=str, choices=["native", "whatsapp", "youtube"])
    parser.add_argument('--global_results_dir', type=Path, required=True, help='Path to results dir')
    parser.add_argument('--const_type', type=none_or_str, default=None, help='Constraint type')
    parser.add_argument('--net_type', type=str, default='mobile',
                        choices=['mobile', 'eff', 'misl', 'mobile_supcon', 'resnet_supcon', 'eff_supcon'])
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--homo_or_not', type=none_or_bool, default=None,
                        help='Filter dataset by homogeneous frames if `True`, or'
                             'Filter dataset by non-homogeneous frames if `False`, or '
                             'Do not perform any filtering - set to `None`')
    p = parser.parse_args()
    print(f'Homogeneous dataset option - {p.homo_or_not}')
    print(f'Using pre-trained model - {p.use_pretrained}')

    if p.gpu_id is not None:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[p.gpu_id], device_type='GPU')

    return p


def run_flow():
    p = parse_args()

    data_factory = DataFactory(input_dir=p.dataset,
                               batch_size=p.batch_size,
                               height=p.height,
                               width=p.width,
                               frames_per_video = p.frames_per_video,
                               homogeneous_frames=p.homo_or_not)

    distance_matrix = None  # data_factory.get_distance_matrix()
    num_classes = len(data_factory.get_class_names())
    train_ds, num_batches = data_factory.get_tf_train_data(category=p.category)
    _, val_ds = data_factory.get_tf_val_data(category=p.category)

    if p.net_type == 'mobile':
        net = MobileNet(num_batches, p.global_results_dir, p.const_type, lr=p.lr)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
        else:
            net.create_model(num_classes, p.height, p.width, distance_matrix, p.model_name, p.use_pretrained)

    elif p.net_type == 'eff':
        net = EfficientNet(num_batches, p.global_results_dir, p.const_type)
        net.create_model(num_classes, p.height, p.width, p.model_name, p.use_pretrained)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)

    elif p.net_type == 'misl':
        net = MISLNet(num_batches, p.global_results_dir, p.const_type)
        net.create_model(num_classes, p.fc_layers, p.fc_size, p.height, p.width, p.model_name)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)

    elif p.net_type == 'mobile_supcon':
        net = MobileNetContrastive(num_batches, p.global_results_dir, p.const_type, lr=p.lr)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
        else:
            net.create_model(num_classes, p.height, p.width, 'encoder', p.model_name + '_enc', p.use_pretrained)
            net.print_model_summary()
            net.train(train_ds=train_ds, val_ds=val_ds, epochs=p.epochs)
            net.create_model(num_classes, p.height, p.width, 'classifier', p.model_name, p.use_pretrained)

    elif p.net_type == 'resnet_supcon':
        net = ResNetContrastive(num_batches, p.global_results_dir, p.const_type, lr=p.lr)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
            net.encoder = tf.keras.Model(net.model.input, net.model.layers[-2].output)
            net.create_model(num_classes, p.height, p.width, 'classifier', p.model_name, p.use_pretrained)
        else:
            net.create_model(num_classes, p.height, p.width, 'encoder', p.model_name + '_enc', p.use_pretrained)
            net.print_model_summary()
            net.train(train_ds=train_ds, val_ds=val_ds, epochs=p.epochs)
            net.create_model(num_classes, p.height, p.width, 'classifier', p.model_name, p.use_pretrained)

    elif p.net_type == 'eff_supcon':
        net = EfficientNetB0Contrastive(num_batches, p.global_results_dir, p.const_type, lr=p.lr)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
            net.encoder = tf.keras.Model(net.model.input, net.model.layers[-2].output)
            net.create_model(num_classes, p.height, p.width, 'classifier', p.model_name, p.use_pretrained)
        else:
            net.create_model(num_classes, p.height, p.width, 'encoder', p.model_name + '_enc', p.use_pretrained)
            net.print_model_summary()
            net.train(train_ds=train_ds, val_ds=val_ds, epochs=p.epochs)
            net.create_model(num_classes, p.height, p.width, 'classifier', p.model_name, p.use_pretrained)

    else:
        raise ValueError('Invalid net type')

    net.print_model_summary()
    net.train(train_ds=train_ds, val_ds=val_ds, epochs=p.epochs)


if __name__ == "__main__":
    run_flow()
