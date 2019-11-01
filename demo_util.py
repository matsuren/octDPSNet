#!/usr/bin/env python
# coding: utf-8
import os
# limit number of thread for Open3D
os.environ["OMP_NUM_THREADS"] = "2"
# show pointclouds
import open3d as o3d
import numpy as np

import torch

import cv2
import matplotlib.pyplot as plt

def ToImg(img):
    return img.cpu().numpy().transpose(1,2,0)/2 + 0.5

def showImg(img):
    ret = ToImg(img)
    plt.imshow(ret)
    
def imread(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def ToT(img):
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()/255
    return (img-0.5)*2

def preprocess(tgt_img, imgs, ref_poses, intrinsics):
    tgt_img = ToT(tgt_img)[np.newaxis]
    imgs = [ToT(it)[np.newaxis] for it in imgs]
    ref_poses = [torch.from_numpy(it)[np.newaxis].float() for it in ref_poses]
    inv_intrinsic = np.linalg.inv(intrinsics)
    intrinsics = torch.from_numpy(intrinsics).float()
    inv_intrinsic = torch.from_numpy(inv_intrinsic).float()
    return tgt_img, imgs, ref_poses, intrinsics, inv_intrinsic

def loadOpenVSLAMpose(fname):
    # cam to world
    pose = np.loadtxt(fname).reshape(4,4)
    # world to cam
    pose = np.linalg.inv(pose)
    return pose

def loadOpenVSLAMcamera(fname):
    cam = np.loadtxt(fname)
    if cam.shape[0] != 9:
        Exception('Error! Fisheye model')

    K = np.eye(3)
    D = np.zeros((5, 1))

    K[0][0] = cam[0]
    K[1][1] = cam[1]
    K[0][2] = cam[2]
    K[1][2] = cam[3]
    D[:] = cam[4:, np.newaxis]
    return K, D

def resizeImgK(imgs, intrinsics, size_xy=(640, 480)):
    resize_x = size_xy[0]
    resize_y = size_xy[1]
    resize_size = (resize_x, resize_y)
    h, w, c = imgs[0].shape
    x_scale = resize_x/w
    y_scale = resize_y/h
    intrinsics[0] *= x_scale
    intrinsics[1] *= y_scale
    
    imgs = [cv2.resize(it, resize_size) for it in imgs]
    return imgs, intrinsics

def getOpen3dCam(img, K):
    height, width, _ = img.shape
    cam = K
    fx = cam[0,0]
    fy = cam[1,1]
    cx = cam[0,2]
    cy = cam[1,2]
    cam_o3 = o3d.camera.PinholeCameraIntrinsic(width, height, fx,fy,cx,cy)
    return cam_o3

def pcdCamera(scale=1.0, color=[1, 0, 0]):
    points = [[-1, 0.75, 0],[1, 0.75, 0],[1, -0.75, 0],[-1, -0.75, 0],[0,0,0.6]]
    lines = [[0,1],[1,2],[2,3],[3,0],
             [0,4],[1,4],[2,4],[3,4]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()

    points = np.array(points)*scale
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def showDepthandPose(img, depth, K, ref_poses, size_xy=(640,480)):
    
    if isinstance(ref_poses, torch.Tensor):
        ref_poses = ref_poses.cpu().numpy()[0]
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()[0]
        
    colors = [[i==0, i==1, i==2] for i in range(3)]
    while len(colors) < len(ref_poses)+1:
        colors.append(np.random.random(3))
        
    resize_x = size_xy[0]
    resize_y = size_xy[1]
    img = cv2.resize(img, (resize_x, resize_y))
    assert img.shape[:2] == depth.shape[:2]

    height, width, _ = img.shape
    cam = K
    fx = cam[0,0]
    fy = cam[1,1]
    cx = cam[0,2]
    cy = cam[1,2]
    cam_o3 = o3d.camera.PinholeCameraIntrinsic(width, height, fx,fy,cx,cy)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(img), o3d.geometry.Image(depth), 
        depth_scale=1.0, depth_trunc=100, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_o3)

    d = depth
    camera_scale = d[d>0.01].min()/9
    cam_ext = ref_poses
    cam_pcds = []
    cam_pcd = pcdCamera(-camera_scale, colors[0])
    cam_pcds.append(cam_pcd)

    for i, it in enumerate(cam_ext):
        cam_pcd = pcdCamera(-camera_scale, colors[i+1])
        T = np.concatenate((it, np.array([[0,0,0,1]])), axis=0)
        invT = np.linalg.inv(T)
        cam_pcd.transform(invT)
        cam_pcds.append(cam_pcd)

    show_pcds = [pcd, *cam_pcds]
    for it in show_pcds:
        it.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries(show_pcds)
