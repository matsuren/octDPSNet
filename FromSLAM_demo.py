# # Demo for OctDPSNet and OpenVSLAM
import os
from os.path import join
from os.path import exists

# limit number of thread for Open3D
os.environ["OMP_NUM_THREADS"] = "2"
import open3d as o3d
from models.octDPSNet import octdpsnet
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models import octconv
import cv2
import logging
import matplotlib.pyplot as plt
from functools import partial
from numpy.linalg import inv
from demo_util import *

from volume_viewer import TSDFVolumeViewer
from glob import glob
from time import sleep

import tkinter as tk
from tkinter import filedialog

# global
mymodel = None
folder_path = None
size_xy = None


def multiViewDepth(mymodel, folder, idx, img_prefix='seq_img', size_xy=(640, 480), VIS=False):
    print(idx)
    fnames = []
    imgs = []
    for i in idx:
        fname = '{}{:03}.jpg'.format(img_prefix, i)
        fnames.append(join(folder, fname))
        imgs.append(imread(fnames[-1]))

    K, D = loadOpenVSLAMcamera(join(folder, 'camera.txt'))

    img_size = (imgs[0].shape[1], imgs[0].shape[0])
    newK = K
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), newK, img_size, cv2.CV_16SC2)

    remap = partial(cv2.remap, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgs = [remap(it, map1, map2) for it in imgs]

    intrinsics = newK
    poses = [loadOpenVSLAMpose(it.replace('.jpg', '.txt')) for it in fnames]

    ref_poses = []
    for i in range(1, len(poses)):
        rel = poses[i].dot(inv(poses[0]))[np.newaxis, :3, :].astype(np.float32)
        ref_poses.append(rel)

    # preprocess image
    imgs, intrinsics = resizeImgK(imgs, intrinsics, size_xy)
    # visualize
    if VIS:
        subtitle = ['left image', 'center image (Keyframe)', 'right image']
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax = ax.flatten()
        for i, it in enumerate(imgs):
            ax[i].imshow(imgs[i])
            ax[i].set_title(subtitle[i])

    org_img = np.array(imgs[0], dtype=np.uint8)
    tgt_img, imgs, ref_poses, intrinsics, inv_intrinsic = preprocess(imgs[0], imgs[1:], ref_poses, intrinsics)
    with torch.no_grad():
        tgt_img = tgt_img.cuda()
        imgs = [it.cuda() for it in imgs]
        ref_poses = [it.cuda() for it in ref_poses]
        intrinsics = intrinsics[np.newaxis].cuda()
        inv_intrinsic = inv_intrinsic[np.newaxis].cuda()
        ref_poses = torch.cat(ref_poses, 1)
        output_depth = mymodel(tgt_img, imgs, ref_poses, intrinsics, inv_intrinsic)
    depth = output_depth.squeeze().cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()[0]
    ref_poses = ref_poses.cpu().numpy()[0]

    if VIS:
        print('ref pose')
        for it in ref_poses:
            print(it)
        print('depth max:', depth.max())
        print('depth min:', depth.min())
        plt.figure()
        plt.title('Depth')
        plt.imshow(depth)

    return depth, org_img, intrinsics, ref_poses


def three_view():
    global mymodel, folder_path, size_xy
    folder = folder_path.get()
    idx = [1, 0, 2]
    depth, org_img, intrinsics, ref_poses = multiViewDepth(mymodel, folder, idx, img_prefix='img', size_xy=size_xy)
    showDepthandPose(org_img, depth, intrinsics, ref_poses, size_xy)
    print('depth max:', depth.max())
    print('depth min:', depth.min())


def volume_reconstruction():
    global mymodel, folder_path, size_xy
    folder = folder_path.get()
    # idx=[1,0,2]
    # depth, org_img, intrinsics, ref_poses = multiViewDepth(mymodel, folder, idx, size_xy)

    mgn = 10  # margin
    # # Main Loop
    volviewer = TSDFVolumeViewer(voxel_length=8.0 / 512.0, sdf_trunc=0.12)
    volviewer.non_blocking_run()
    # waiting visualization to be ready
    while not volviewer.running:
        sleep(0.1)
    for i in range(1000):
        print('waiting for new images,,,', i)
        if i == 0:
            idx = [0, 1, 2]
        else:
            idx = [i, i - 1, i + 1]

        img_fname = 'seq_img{:03}.jpg'.format(max(idx))
        while not exists(join(folder, img_fname)) and volviewer.running:
            sleep(0.05)
        # check if it's closed
        if not volviewer.running:
            break

        print('detect new images', idx)
        depth, org_img, intrinsics, ref_poses = multiViewDepth(mymodel, folder, idx, 'seq_img', size_xy)

        print('depth generated')
        print('depth:', depth.min(), depth.max())
        depth_crop = np.zeros_like(depth)
        # remove margin
        depth_crop[mgn:-mgn, mgn:-mgn] = depth[mgn:-mgn, mgn:-mgn]
        fname = join(folder, 'seq_img{:03}.jpg'.format(idx[0]))
        T = loadOpenVSLAMpose(fname.replace('.jpg', '.txt'))

        # Update TSDFVolumeViewer
        volviewer.addRGBD(org_img, depth_crop, intrinsics, T)
        volviewer.update_pcd()

    print('End:')


def delete_sequence():
    global folder_path
    folder = folder_path.get()
    print('Delete all image sequence in', folder)
    for it in glob(join(folder, 'seq_img*.jpg')):
        os.remove(it)
        os.remove(it.replace('.jpg', '.txt'))


def main():
    global mymodel, folder_path, size_xy
    gpu_id = 0
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    print('device:{}'.format(device))

    nlabel = 32
    octconv.ALPHA = 0.75
    mindepth = 0.5
    size_xy = (640 // 2, 480 // 2)

    #########################
    ## Please wait window
    #########################
    window = tk.Tk()
    window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file='./img/icon.png'))
    window.title("octDPSNet with OpenVSLAM")
    window.configure(background='#3E3D60')
    text = "Loading model now!\n\nPlease wait for a while\n" + \
           'device:{}\nmodel detail:a={}, nlabel={}, size=({}x{})'.format(
               device, octconv.ALPHA, nlabel, size_xy[0], size_xy[1])

    lbl = tk.Label(window, text=text, fg="white", bd=1, anchor=tk.S, font='Helvetica 12 bold')
    lbl.config(background='#3E3D60')
    lbl.pack(fill='both', padx=30, pady=30)
    window.update()

    #########################
    ## Load model
    #########################
    print('Start loading')
    mymodel = octdpsnet(nlabel, mindepth, octconv.ALPHA, True).to(device)
    mymodel.eval()
    print('Finish loading')
    window.destroy()

    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=formatter)
    logging.basicConfig(level=logging.WARN, format=formatter)

    ############################
    # select data
    ############################
    def browse_button():
        global folder_path, folder
        folder_name = filedialog.askdirectory(initialdir=folder_path)
        if folder_name:
            folder_path.set(folder_path)
            print(folder_path)
        else:
            print('canceled folder dialog')

    window = tk.Tk()
    window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file='./img/icon.png'))
    window.title("octDPSNet with OpenVSLAM")
    # window.geometry('600x200')
    window.configure(background='#3E3D60')

    #########################
    ## folder select
    #########################
    btn_folder = tk.Button(window, text="choose folder", command=browse_button)
    btn_folder.grid(row=2, column=0, padx=16, pady=1)
    folder_path = tk.StringVar(window, value='./fromSLAM/')
    lbl1 = tk.Label(master=window, textvariable=folder_path, relief=tk.SUNKEN)
    lbl1.config(width=40)
    lbl1.grid(row=2, column=1, padx=16, pady=1, columnspan=2)

    #########################
    ## Three view reconstruction
    #########################
    btn_threeview = tk.Button(window, text="Three view", bg="white", command=three_view)
    btn_threeview.config(height=2, width=15, background='#B1B5E0')
    btn_threeview.grid(row=0, column=0, padx=10, pady=16)

    #########################
    ## Volume reconstruction
    #########################
    btn_volume = tk.Button(window, text="Volume reconstruction", bg="white", command=volume_reconstruction)
    btn_volume.config(height=2, width=20, background='#B1B5E0')
    btn_volume.grid(row=0, column=1, padx=10, pady=16)

    #########################
    ## Clear volume folder
    #########################
    btn_volume = tk.Button(window, text="Clear sequence", bg="white", command=delete_sequence)
    btn_volume.config(height=2, width=10, background='#B1B5E0')
    btn_volume.grid(row=0, column=2, padx=10, pady=16)

    #########################
    ## Status bar
    #########################
    statusbar = tk.Label(window, text="Select folder which includes images and poses",
                         fg="white", bd=1, anchor=tk.S, font='Helvetica 12 bold')
    statusbar.grid(row=1, column=0, columnspan=3, padx=16, pady=5, sticky=tk.W + tk.E + tk.S)
    statusbar.config(background='#3E3D60')
    window.mainloop()


if __name__ == "__main__":
    main()
