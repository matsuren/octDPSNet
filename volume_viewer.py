#!/usr/bin/env python
# coding: utf-8
import os
# limit number of thread for Open3D
os.environ["OMP_NUM_THREADS"] = "2"
# show pointclouds
import open3d as o3d
import numpy as np
import logging
from logging import getLogger
import threading
from time import sleep, time


def getOpen3dCam(img, K):
    height, width, _ = img.shape
    cam = K
    fx = cam[0, 0]
    fy = cam[1, 1]
    cx = cam[0, 2]
    cy = cam[1, 2]
    cam_o3 = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    return cam_o3


class TSDFVolumeViewer(object):
    vis = None  # don't create multiple Visualizer

    def __init__(self, voxel_length, sdf_trunc, log_level=logging.DEBUG):
        assert TSDFVolumeViewer.vis is None

        self.volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=voxel_length, sdf_trunc=sdf_trunc,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        #         self.volume = o3d.integration.UniformTSDFVolume(
        #             length=10.0,
        #             resolution=512,
        #             sdf_trunc=0.12,
        #             color_type=o3d.integration.TSDFVolumeColorType.RGB8)

        self.num = 0
        self.show_pcd = None
        self.logger = getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info('create')

        self.stop_thread = False

    def __del__(self):
        self.stop_non_blocking()
        print("Destructor called")

    @property
    def running(self):
        return TSDFVolumeViewer.vis != None

    def addRGBD(self, img, depth, K, T):
        start = time()
        cam_o3 = getOpen3dCam(img, K)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(img), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=100, convert_rgb_to_intensity=False)
        self.volume.integrate(rgbd_image, cam_o3, T)
        self.num += 1
        self.logger.info('Add RGBD images:{}, Elps {:.3}'.format(self.num, time() - start))

    def extract_mesh(self):
        mesh = self.volume.extract_triangle_mesh()
        #         mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return mesh

    def extract_point_cloud(self):
        mesh = self.volume.extract_point_cloud()
        #         mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return mesh

    def update_pcd(self):
        if TSDFVolumeViewer.vis == None:
            self.logger.error('Start non_blocking_run first')
            return
        #         self.logger.info('start update_pcd:')
        start = time()
        if True:
            pcd = self.extract_point_cloud()
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            if self.show_pcd:
                self.show_pcd.colors = pcd.colors
                self.show_pcd.points = pcd.points
            else:
                self.show_pcd = pcd
                self.show_pcd.normals = o3d.utility.Vector3dVector()
                self.vis.add_geometry(self.show_pcd)
        else:
            mesh = self.extract_mesh()
            mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            if self.show_pcd:
                self.show_pcd.vertex_colors = mesh.vertex_colors
                self.show_pcd.vertices = mesh.vertices
                self.show_pcd.triangles = mesh.triangles
            else:
                self.show_pcd = mesh
                self.vis.add_geometry(self.show_pcd)
        self.logger.info('Update pcd: Elps {:.3}'.format(time() - start))
        start = time()

    def non_blocking_run(self):
        if self.running:
            self.logger.info('already runnging')
            return

        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop_non_blocking(self):
        if not self.running:
            self.logger.info('no thread')
            return

        self.stop_thread = True
        self.thread.join()
        self.logger.info('finish stop_non_blocking')

    def run(self):
        self.logger.info('non_blocking_run start')
        TSDFVolumeViewer.vis = o3d.visualization.Visualizer()
        TSDFVolumeViewer.vis.create_window(width=800, height=600)

        while True:
            TSDFVolumeViewer.vis.update_geometry()
            if (not TSDFVolumeViewer.vis.poll_events() or self.stop_thread):
                break
            TSDFVolumeViewer.vis.update_renderer()
            sleep(0.01)  # time for other threads

        self.logger.info('non_blocking_run end')
        TSDFVolumeViewer.vis.destroy_window()
        TSDFVolumeViewer.vis = None