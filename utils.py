from __future__ import division
import numpy as np
import cv2
import torch
from path import Path
import datetime
from collections import OrderedDict


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)

    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path / timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            cmap = getattr(cv2, 'COLORMAP_{}'.format(colormap.upper()))
        except Exception as e:
            print('No colormap:', colormap.upper())
        array = (tensor.squeeze().numpy() / max_value).clip(0, 1)
        array = (array * 255).clip(0, 255).astype(np.uint8)
        colored_array = cv2.applyColorMap(array, cmap)
        array = cv2.cvtColor(colored_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        array = array.transpose(2, 0, 1)
    elif tensor.ndimension() == 3:
        # assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy() * 0.5
    return array



def save_checkpoint(save_path, one_state, epoch, filename='checkpoint.pth.tar'):
    file_prefixes = ['octdpsnet']
    states = [one_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path / '{}_{}_{}'.format(prefix, epoch, filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0] * i
        self.sum = [0] * i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert (len(val) == self.meters)
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n

    @property
    def avg(self):
        return [it / self.count for it in self.sum]

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)
