from models import octDPSNet as PSNet

import argparse
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import custom_transforms
from utils import tensor2array, save_checkpoint, save_path_formatter
from loss_functions import compute_errors_train

from utils import AverageMeter
from tensorboardX import SummaryWriter
from sequence_folders import SequenceFolder
from models import octconv
from radam import RAdam

parser = argparse.ArgumentParser(description='Training for octDPSNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=16, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-output', action='store_true', help='show depth for tensorboard')
parser.add_argument('--ttype', default='train.txt', type=str, help='Text file indicates input data')
parser.add_argument('--ttype2', default='val.txt', type=str, help='Text file indicates input data')
parser.add_argument('-f', '--training-output-freq', type=int,
                    help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)
parser.add_argument('--nlabel', type=int, default=64, help='number of label')
parser.add_argument('--mindepth', type=float, default=0.5, help='minimum depth')
parser.add_argument('--alpha', type=float, default=0.9375,
                    help='ratio of low frequency')  # 0.9375, 0.875, 0.75, 0.5, 0.25
# parser.add_argument('--reduction', type=int, default=8, help='reduction rate for oct SE')  # 8, 16
parser.add_argument('--val-from', type=int, default=0, help='start validation from N epoch')

n_iter = 0
start_epoch = 0


def main():
    global n_iter, start_epoch
    args = parser.parse_args()

    #################################
    # Hyper parameter
    octconv.ALPHA = args.alpha
    #################################
    # save folder
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints' / save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path / 'valid' / str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ColorJitter(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        ttype=args.ttype
    )
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        ttype=args.ttype2
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    octdps = PSNet(args.nlabel, args.mindepth).cuda()
    cudnn.benchmark = True
    if False:
        print('=> setting adam solver')
        optimizer = torch.optim.Adam(octdps.parameters(), args.lr,
                                     betas=(args.momentum, args.beta),
                                     weight_decay=args.weight_decay)
    else:
        print('=> setting radam solver')
        optimizer = RAdam(octdps.parameters(), args.lr, betas=(args.momentum, args.beta),
                          weight_decay=args.weight_decay)

    print('=> setting scheduler')
    scheduler = StepLR(optimizer, step_size=12, gamma=0.1)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        print("=> using pre-trained weights for DPSNet")
        octdps.load_state_dict(checkpoint['state_dict'])

        start_epoch = checkpoint['epoch']
        n_iter = checkpoint['n_iter']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Resume training from epoch {}".format(start_epoch))
        for param_group in optimizer.param_groups:
            print('lr:', param_group['lr'])
    else:
        octdps.init_weights()
    octdps = torch.nn.DataParallel(octdps)

    with open(args.save_path / args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['epoch', 'train_loss', 'validation_loss'])

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_loss = train(args, train_loader, octdps, optimizer, args.epoch_size, training_writer, epoch)

        if args.val_from <= epoch:
            errors, error_names = validate_with_gt(args, val_loader, octdps, epoch, output_writers)
            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, epoch)
            # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            decisive_error = errors[0]
        else:
            decisive_error = -1
        scheduler.step()
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'n_iter': n_iter,
                'decisive_error': decisive_error,
                'state_dict': octdps.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            epoch)

        with open(args.save_path / args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epoch, train_loss, decisive_error])


def train(args, train_loader, octdps, optimizer, epoch_size, train_writer, epoch):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    # switch to train mode
    octdps.train()

    end = time.time()

    for i, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = tgt_img.cuda()
        ref_imgs_var = [img.cuda() for img in ref_imgs]
        ref_poses_var = [pose.cuda() for pose in ref_poses]
        intrinsics_var = intrinsics.cuda()
        intrinsics_inv_var = intrinsics_inv.cuda()
        tgt_depth_var = tgt_depth.cuda()

        # compute output
        pose = torch.cat(ref_poses_var, 1)

        # get mask
        mask = (tgt_depth_var <= args.nlabel * args.mindepth) & (tgt_depth_var >= args.mindepth)
        mask.detach_()

        #
        depths = octdps(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
        disps = [args.mindepth * args.nlabel / depth for depth in depths]

        loss = 0.
        # loss += F.smooth_l1_loss(torch.squeeze(depths[0], 1)[mask], tgt_depth_var[mask], reduction='mean') * 0.5
        loss += F.smooth_l1_loss(torch.squeeze(depths[1], 1)[mask], tgt_depth_var[mask], reduction='mean') * 1
        loss += F.smooth_l1_loss(torch.squeeze(depths[2], 1)[mask], tgt_depth_var[mask], reduction='mean') * 0.7

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('total_loss', loss.item(), n_iter)
            for _tmp, param_group in enumerate(optimizer.param_groups):
                train_writer.add_scalar('learning_rate{}'.format(_tmp), param_group['lr'], n_iter)

        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

            train_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)

            depth_to_show = tgt_depth[0]
            disp_to_show = (args.nlabel * args.mindepth / depth_to_show)
            disp_to_show[disp_to_show > args.nlabel] = 0
            train_writer.add_image('train Dispnet GT Normalized',
                                   tensor2array(disp_to_show, max_value=args.nlabel, colormap='bone'),
                                   n_iter)
            train_writer.add_image('train Depth GT Normalized',
                                   tensor2array(depth_to_show, max_value=args.nlabel * args.mindepth * 0.3),
                                   n_iter)

            for k, scaled_depth in enumerate(depths):
                train_writer.add_image('train Dispnet Output Normalized {}'.format(k),
                                       tensor2array(disps[k].detach()[0].cpu(), max_value=args.nlabel, colormap='bone'),
                                       n_iter)
                train_writer.add_image('train Depth Output Normalized {}'.format(k),
                                       tensor2array(depths[k].detach()[0].cpu(),
                                                    max_value=args.nlabel * args.mindepth * 0.3),
                                       n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Train({}): Time {} Data {} Loss {}, {}/{}({:.2f}%)'.format(
                epoch, batch_time, data_time, losses, i, len(train_loader), 100 * i / len(train_loader)), flush=True)

        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def validate_with_gt(args, val_loader, octdps, epoch, output_writers=[]):
    batch_time = AverageMeter()
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    octdps.eval()

    end = time.time()
    with torch.no_grad():
        for i, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth) in enumerate(val_loader):
            tgt_img_var = tgt_img.cuda()
            ref_imgs_var = [img.cuda() for img in ref_imgs]
            ref_poses_var = [pose.cuda() for pose in ref_poses]
            intrinsics_var = intrinsics.cuda()
            intrinsics_inv_var = intrinsics_inv.cuda()
            tgt_depth_var = tgt_depth.cuda()

            pose = torch.cat(ref_poses_var, 1)

            output_depth = octdps(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
            output_disp = args.nlabel * args.mindepth / (output_depth)

            mask = (tgt_depth <= args.nlabel * args.mindepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)

            output = torch.squeeze(output_depth.data.cpu(), 1)

            if log_outputs and i % 100 == 0 and i / 100 < len(output_writers):
                index = int(i // 100)
                if epoch == 0:
                    output_writers[index].add_image('val Input', tensor2array(tgt_img[0]), 0)
                    depth_to_show = tgt_depth_var.data[0].cpu()
                    depth_to_show[depth_to_show > args.nlabel * args.mindepth] = args.nlabel * args.mindepth
                    disp_to_show = (args.nlabel * args.mindepth / depth_to_show)
                    disp_to_show[disp_to_show > args.nlabel] = 0

                    output_writers[index].add_image('val target Disparity Normalized',
                                                    tensor2array(disp_to_show, max_value=args.nlabel, colormap='bone'),
                                                    epoch)
                    output_writers[index].add_image('val target Depth Normalized', tensor2array(depth_to_show,
                                                                                                max_value=args.nlabel * args.mindepth * 0.3),
                                                    epoch)

                output_writers[index].add_image('val Dispnet Output Normalized',
                                                tensor2array(output_disp.data[0].cpu(), max_value=args.nlabel,
                                                             colormap='bone'), epoch)
                output_writers[index].add_image('val Depth Output', tensor2array(output_depth.data[0].cpu(),
                                                                                 max_value=args.nlabel * args.mindepth * 0.3),
                                                epoch)

            errors.update(compute_errors_train(tgt_depth, output, mask))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('valid({}): Time {} Abs Error {:.4f} ({:.4f}), {}/{}({:.2f}%)'.format(
                    epoch, batch_time, errors.val[0], errors.avg[0], i, len(val_loader), 100 * i / len(val_loader)),
                    flush=True)

    return errors.avg, error_names


if __name__ == '__main__':
    main()
