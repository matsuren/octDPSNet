from models.octDPSNet import octdpsnet

import argparse
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
from utils import tensor2array
from loss_functions import compute_errors_test
from sequence_folders import SequenceFolder
import matplotlib.pyplot as plt
from models import octconv

import os
from path import Path
from imageio import imwrite

import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


parser = argparse.ArgumentParser(description='Test octDPSNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for testing', default=2)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--pretrained', dest='pretrained', default=None, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--output-dir', default='results', type=str, help='Output directory')
# parser.add_argument('--ttype', default='test.txt', type=str, help='Text file indicates input data')
parser.add_argument('--nlabel', type=int, default=64, help='number of label')
parser.add_argument('--mindepth', type=float, default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float, default=10, help='maximum depth')
parser.add_argument('--alpha', type=float, default=0.9375,
                    help='ratio of low frequency')  # 0.9375, 0.875, 0.75, 0.5, 0.25


# parser.add_argument('--reduction', type=int, default=8, help='reduction rate for oct SE')  # 8, 16

def generateDataset_test(FOLDER):
    # Generate dataset_test.txt
    with open(FOLDER + '/test.txt') as f:
        data = f.readlines()

    id_range = []
    dataset_names = []
    dataset_name = None
    start = -1
    for i, it in enumerate(data):
        if dataset_name == it[:it.find('_')]:
            continue
        else:
            if start < 0:
                dataset_name = it[:it.find('_')]
                start = i
            else:
                dataset_names.append(dataset_name)
                id_range.append([start, i])
                start = i
                dataset_name = it[:it.find('_')]
    dataset_names.append(dataset_name)
    id_range.append([start, i])

    print('Gerenate dataset_test.txt from test.txt')
    print('Datasets:', dataset_names)
    print('id_range:', id_range)

    # Save data
    for i in range(len(dataset_names)):
        filename = FOLDER + '/{}_test.txt'.format(dataset_names[i])
        it = id_range[i]
        with open(filename, mode='w') as f:
            f.writelines(data[it[0]:it[1]])


def main():
    args = parser.parse_args()

    ttypes = ['mvs_test.txt', 'sun3d_test.txt', 'rgbd_test.txt', 'scenes11_test.txt']
    # generate ttypes from test.txt
    if not os.path.exists(os.path.join(args.data, ttypes[0])):
        generateDataset_test(args.data)

    #################################
    # Hyper parameter
    octconv.ALPHA = args.alpha
    #################################

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        ttype='test.txt',
        sequence_length=args.sequence_length
    )

    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    cudnn.benchmark = True

    #################################
    # Model
    #################################
    if args.pretrained:
        octdps = octdpsnet(args.nlabel, args.mindepth, args.alpha, False).cuda()
        weights = torch.load(args.pretrained)
        pretrained_name = args.pretrained.split('/')[-1].split('.')[0]
        output_dir = Path(args.output_dir + '_' + pretrained_name)
        octdps.load_state_dict(weights['state_dict'])
    else:
        print('load pretrained model from internet')
        octdps = octdpsnet(args.nlabel, args.mindepth, args.alpha, True).cuda()
        output_dir = Path(args.output_dir + '_' + 'octdps{}n{}'.format(int(100 * octconv.ALPHA), args.nlabel))

    octdps.eval()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print(output_dir)

    # save all error to analyze later
    save_depth_error = []
    save_elps = []
    all_cnt = 0
    print("{}".format(args.output_dir))
    errors_all = OrderedDict({})

    for ttype in ttypes:
        valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            ttype=ttype,
            sequence_length=args.sequence_length
        )

        dataset_name = ttype.split('_')[0]
        print('dataset:{}'.format(dataset_name))
        print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        errors = np.zeros((2, 8, len(val_loader)), np.float32)
        for ii, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth) in enumerate(val_loader):
            with torch.no_grad():
                i = ii
                tgt_img_var = tgt_img.cuda()
                ref_imgs_var = [img.cuda() for img in ref_imgs]
                ref_poses_var = [pose.cuda() for pose in ref_poses]
                intrinsics_var = intrinsics.cuda()
                intrinsics_inv_var = intrinsics_inv.cuda()

                # compute output
                pose = torch.cat(ref_poses_var, 1)
                start = time.time()
                output_depth = octdps(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
                elps = time.time() - start
                save_elps.append(elps)
                tgt_disp = args.mindepth * args.nlabel / tgt_depth
                output_disp = args.mindepth * args.nlabel / output_depth

                mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)

                output_disp_ = torch.squeeze(output_disp.data.cpu(), 1)
                output_depth_ = torch.squeeze(output_depth.data.cpu(), 1)

                errors[0, :, i] = compute_errors_test(tgt_depth[mask], output_depth_[mask])
                errors[1, :, i] = compute_errors_test(tgt_disp[mask], output_disp_[mask])

                print('Elapsed Time {} Abs Error {:.4f}'.format(elps, errors[0, 0, i]))

                save_depth_error.append(
                    (errors[0, 0, i], output_depth_.numpy().squeeze(), tgt_depth.numpy().squeeze(),
                     tgt_img.numpy(), ref_imgs[0].numpy(), all_cnt))
                output_disp_n = (output_disp_).numpy()[0]
                np.save(output_dir / '{:04d}{}'.format(all_cnt, '.npy'), output_disp_n)

                disp = (255 * tensor2array(torch.from_numpy(output_disp_n),
                                           max_value=args.nlabel, colormap='bone')).astype(np.uint8)
                imwrite(output_dir / '{:04d}_disp{}'.format(all_cnt, '.png'), disp.transpose(1, 2, 0))
                all_cnt += 1

        errors_all[dataset_name] = errors
        mean_errors = errors.mean(2)

        error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']

        print("Depth Results : ")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

        print("Disparity Results : ")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))

    with open(output_dir / 'errors_all.json', 'w') as f:
        data = {'header': args.pretrained, 'errors_all': errors_all}
        f.write(json.dumps(data, cls=NumpyEncoder))

    # print and save results
    print('summary results')
    print_array = []
    print_array.append("Depth Results : ")
    print_array.append("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))

    concat = []
    for key, val in errors_all.items():
        print_array.append('dataset:{}'.format(key))
        mean_errors = val.mean(2)
        concat.append(val)
        print_array.append(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    # ALL
    print_array.append('dataset:ALL')
    mean_errors = np.concatenate(concat, axis=2).mean(2)
    print_array.append(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))
    print_array.append("Average Elapsed time:{:6.4f}".format(np.array(save_elps).mean()))

    with open(output_dir / 'summary_results.txt', mode='w') as f:
        print(args.pretrained, file=f)
        for it in print_array:
            print(it)
            print(it, file=f)

    # # Visualization
    # def toImg(img):
    #     ret = img[0].transpose(1, 2, 0)
    #     return ret / 2 + 0.5
    #
    # sorted_error = sorted(save_depth_error, reverse=True)
    # # save big error pictures and small error pictures
    # for tmp in range(1, 15):
    #     it = sorted_error[-tmp]
    #     fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    #     fig.suptitle("{}: Abs error: {:.4f}, id:{}".format(tmp, it[0], it[5]), fontsize=14)
    #     plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.93, hspace=0.1, wspace=0.1)
    #     ax = ax.flatten()
    #     for i in range(4):
    #         if i < 2:
    #             ax[i].imshow(it[i + 1])
    #         else:
    #             ax[i].imshow(toImg(it[i + 1]))
    #     # Save the full figure...
    #     fname = str(output_dir / 'small_error_pair{}.png'.format(tmp))
    #     fig.savefig(fname)
    #
    # for tmp in range(30):
    #     it = sorted_error[tmp]
    #     fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    #     fig.suptitle("{}: Abs error: {:.4f}, id:{}".format(tmp, it[0], it[5]), fontsize=14)
    #     plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.93, hspace=0.1, wspace=0.1)
    #     ax = ax.flatten()
    #     for i in range(4):
    #         if i < 2:
    #             ax[i].imshow(it[i + 1])
    #         else:
    #             ax[i].imshow(toImg(it[i + 1]))
    #     # Save the full figure...
    #     fname = str(output_dir / 'big_error_pair{}.png'.format(tmp))
    #     fig.savefig(fname)


if __name__ == '__main__':
    # main
    main()
