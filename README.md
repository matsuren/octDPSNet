# OctDPSNet
PyTorch implementation of [Octave Deep Plane-sweeping Network: Reducing Spatial Redundancy for Learning-based Plane-sweeping Stereo](http://dx.doi.org/10.1109/ACCESS.2019.2947195). 

[![octdpsnet_video](img/overlay.png)](http://ieeetv.ieee.org/media/ieeetvmobile/xplore/access-gavideo-2947195.mp4)

If you use OctDPSNet for your academic research, please cite the following paper.
```
  @article{octDPSNet2019,
    author={R. Komatsu, H. Fujii, Y. Tamura, A. Yamashita, and H. Asama},
    journal={IEEE Access},
    title={Octave Deep Plane-sweeping Network: Reducing Spatial Redundancy for Learning-based Plane-sweeping Stereo},
    year={2019},
    volume={7},
    pages={150306-150317},
    doi={10.1109/ACCESS.2019.2947195},
  }
```

## Demo
You can try octDPSNet in Google Colab [here (octDPSNet_demo_colab.ipynb)](https://colab.research.google.com/github/matsuren/octDPSNet/blob/master/octDPSNet_demo_colab.ipynb).

After running all cells, point cloud `results.ply` will be downloaded. You can visualize it by some tools e.g. [Meshlab](http://www.meshlab.net/).

Examples of the point cloud are displayed here.

<span style="display:block;text-align:center">
<img src="./sample_data/mvs_test_00023/out.gif" width="350px">
<img src="./sample_data/sun3d_test_00047/out.gif" width="350px">
</span>

## Requirements
python >= 3.5  
CUDA   
pipenv  

## Install
We recommend you to use `pipenv` to install the correct version of the python libraries since some libraries (e.g. scipy) changed their API which causes some errors. We might modify our source code later to get it working on the latest version of the libraries.

### Install pipenv
```bash
pip install pipenv
```
### Setup virtual environment
```bash
git clone https://github.com/matsuren/octDPSNet.git
cd octDPSNet
pipenv install --dev
```

Since PyTorch is not included in Pipfile, you need to install PyTorch in the virtual environment via pip.
The official instruction is available [here (latest)](https://pytorch.org/get-started/locally/) or
 [here (previous-versions)](https://pytorch.org/get-started/previous-versions/#via-pip). 
Please install PyTorch according to your CUDA version and python version. 

If you're using CUDA 10.0 and Python 3.5, run the following commands to install PyTorch 1.3.0.

```bash
# Enter the virtual environment 
pipenv shell
# Install PyTorch via pip
pip install https://download.pytorch.org/whl/cu100/torch-1.3.0%2Bcu100-cp35-cp35m-linux_x86_64.whl
```

### Verify your installation
Run the following command to make sure that you've installed octDPSNet correctly.
```bash
# Make sure you're in the virtual environment
python demo.py
```
After running the command, you will see point cloud in Open3D visualization window and `results.ply` will be generated in the current directory.

## Combine with OpenVSLAM
### Demo
Some images and camera poses are already bundled with this repository, so you can give it a try even without OpenVSLAM. Run the following command.
```bash
# Make sure you're in the virtual environment
python FromSLAM_demo.py
```

Then, simple GUI with three buttons will be appeared. 

If you want to try depth reconstruction from three view, push `Three view`. 

If you want to try volume reconstruction from multiple view, push `Volume reconstruction`. Depthmaps from three view are integrated by TSDF volume.

### OpenVSLAM
Clone the following repository which is the modified version of OpenVSLAM for OctDPSNet. We just add additional buttons and key binding to Pangolin GUI to save images and camera poses.

```bash
git clone -b octDPSNet https://github.com/matsuren/openvslam.git
```
Please follow the OpenVSLAM official instruction [here](https://openvslam.readthedocs.io/en/master/installation.html) to install OpenVSLAM. 

The explaination of the additional buttons and key bindings are the following.

- Three buttons on Pangolin GUI: `Add left image`, `Add center image`, and `Add right image`, are used to save images and poses for `Three view`. 
- 'S' key is used to save images and poses for `Volume reconstruction`. 

Please open issues if you have any questions.


