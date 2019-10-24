from models.octDPSNet import octdpsnet

import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
from utils import tensor2array
from loss_functions import compute_errors_test
from sequence_folderse import SequenceFolder
from models import octconv

import os
from path import Path
from scipy.misc import imsave

parser = argparse.ArgumentParser(description='ETH dataset test',
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
parser.add_argument('--output-dir', default='resultETH', type=str, help='Output directory')
parser.add_argument('--nlabel', type=int, default=64, help='number of label')
parser.add_argument('--mindepth', type=float, default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float, default=72, help='maximum depth')
parser.add_argument('--output-print', action='store_true', help='print output depth')
parser.add_argument('--alpha', type=float, default=0.9375,
                    help='ratio of low frequency')  # 0.9375, 0.875, 0.75, 0.5, 0.25

cudnn.benchmark = True


def main():
    args = parser.parse_args()
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
        sequence_length=args.sequence_length
    )

    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

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

    errors = np.zeros((2, 8, len(val_loader)), np.float32)
    with torch.no_grad():
        for ii, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, scale_) in enumerate(val_loader):
            i = ii
            tgt_img_var = tgt_img.cuda()
            ref_imgs_var = [img.cuda() for img in ref_imgs]
            ref_poses_var = [pose.cuda() for pose in ref_poses]
            intrinsics_var = intrinsics.cuda()
            intrinsics_inv_var = intrinsics_inv.cuda()
            tgt_depth_var = tgt_depth.cuda()
            scale = scale_.numpy()[0]

            # compute output
            pose = torch.cat(ref_poses_var, 1)
            start = time.time()
            output_depth = octdps(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
            elps = time.time() - start
            mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)

            tgt_disp = args.mindepth * args.nlabel / tgt_depth
            output_disp = args.mindepth * args.nlabel / output_depth

            output_disp_ = torch.squeeze(output_disp.data.cpu(), 1)
            output_depth_ = torch.squeeze(output_depth.data.cpu(), 1)

            errors[0, :, i] = compute_errors_test(tgt_depth[mask] / scale, output_depth_[mask] / scale)
            errors[1, :, i] = compute_errors_test(tgt_disp[mask] / scale, output_disp_[mask] / scale)

            print('Elapsed Time {} Abs Error {:.4f}'.format(elps, errors[0, 0, i]))

            if args.output_print:
                output_disp_n = (output_disp_).numpy()[0]
                np.save(output_dir / '{:04d}{}'.format(i, '.npy'), output_disp_n)
                disp = (255 * tensor2array(torch.from_numpy(output_disp_n), max_value=args.nlabel,
                                           colormap='bone')).astype(np.uint8)
                imsave(output_dir / '{:04d}_disp{}'.format(i, '.png'), disp.transpose(1, 2, 0))

    mean_errors = errors.mean(2)
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']
    print("{}".format(args.output_dir))
    print("Depth Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    # print and save results
    print('summary results')
    print_array = []
    print_array.append("Depth Results ETH3D: ")
    print_array.append("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print_array.append(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    with open(output_dir / 'summary_results.txt', mode='w') as f:
        print(args.pretrained, file=f)
        for it in print_array:
            print(it)
            print(it, file=f)

    np.savetxt(output_dir / 'errors.csv', mean_errors, fmt='%1.4f', delimiter=',')


if __name__ == '__main__':
    main()
