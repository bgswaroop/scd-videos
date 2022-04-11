import argparse
from pathlib import Path

import tensorflow as tf

tf.config.run_functions_eagerly(True)

import dataset
from models import (MISLNet, MobileNet, EfficientNet, ResNet, MobileNetContrastive, ResNetContrastive,
                    EfficientNetB0Contrastive)


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

    # Dataset params
    parser.add_argument('--dataset_name', type=str, choices=['vision', 'qufvd'])
    parser.add_argument('--all_I_frames_dir', type=Path, help='Input directory of extracted I frames')
    parser.add_argument('--all_frames_dir', type=Path, help='Input directory of extracted frames')
    parser.add_argument('--frame_selection', type=str, default='equally_spaced', choices=['equally_spaced', 'first_N'])
    parser.add_argument('--frame_type', type=str, default='I', choices=['I', 'all'])
    parser.add_argument('--fpv', type=int, default=50, help='max number of frames per video (set -1 for all frames)')
    parser.add_argument('--category', type=str, choices=["native", "whatsapp", "youtube"])

    # ConvNet params
    parser.add_argument('--const_type', type=none_or_str, default=None, help='Constraint type')
    parser.add_argument('--model_name', type=str, help='Name for the model')
    parser.add_argument('--use_pretrained', type=none_or_bool, default=True, help='Use pretrained net from ImageNet')
    parser.add_argument('--model_path', type=str, help='Path to model to continue training (*.h5)')
    parser.add_argument('--height', type=int, default=480, help='Height of CNN input dimension [default: 480]')
    parser.add_argument('--width', type=int, default=800, help='Width of CNN input dimension [default: 800]')
    parser.add_argument('--gpu_id', type=int, default=None, help='Choose the available GPU devices')
    parser.add_argument('--global_results_dir', type=Path, required=True, help='Path to results dir')
    parser.add_argument('--const_type', type=none_or_str, default=None, help='Constraint type')
    parser.add_argument('--net_type', type=str, default='mobile',
                        choices=['mobile', 'eff', 'misl', 'res', 'mobile_supcon', 'resnet_supcon', 'eff_supcon'])

    # Optimization params
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')

    # Session params
    parser.add_argument('--gpu_id', type=int, default=None, help='Choose the available GPU devices')
    parser.add_argument('--global_results_dir', type=Path, required=True, help='Path to results dir')

    p = parser.parse_args()
    print(f'Using pre-trained model - {p.use_pretrained}')

    if p.gpu_id is not None:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[p.gpu_id], device_type='GPU')

    return p


def run_flow():
    p = parse_args()

    if p.dataset_name == 'vision':
        data_factory = dataset.vision.DataFactory(p)
    elif p.dataset_name == 'qufvd':
        data_factory = dataset.qufvd.DataFactory(p)
    else:
        raise ValueError(f'Invalid option {p.dataset_name}')

    distance_matrix = None  # data_factory.get_distance_matrix()
    num_classes = len(data_factory.class_names)
    train_ds, num_batches = data_factory.get_tf_train_data(category=p.category)
    filename_ds, val_ds = data_factory.get_tf_val_data(category=p.category)

    if p.net_type == 'mobile':
        net = MobileNet(num_batches, p.global_results_dir, p.const_type, lr=p.lr)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
            net.compile()
        else:
            net.create_model(num_classes, p.height, p.width, distance_matrix, p.model_name, p.use_pretrained)

    elif p.net_type == 'eff':
        net = EfficientNet(num_batches, p.global_results_dir, p.const_type)
        net.create_model(num_classes, p.height, p.width, p.model_name, p.use_pretrained)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
            net.compile()

    elif p.net_type == 'misl':
        net = MISLNet(num_batches, p.global_results_dir, p.const_type)
        net.create_model(num_classes, p.height, p.width, p.model_name)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
            net.compile()

    elif p.net_type == 'res':
        net = ResNet(num_batches, p.global_results_dir, p.const_type)
        net.create_model(num_classes, p.height, p.width, p.model_name, p.use_pretrained)
        if p.model_path:  # to continue the training
            net.set_model(p.model_path)
            net.compile()

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
