import os
import sys
import pathlib
parentFolder = pathlib.Path(__file__).parent.parent.resolve()
parentFolder = os.path.abspath(parentFolder)
sys.path.append(parentFolder)

from fastmvsnet.utils.io import *
import cv2

from picutils import generatePointCloud, savePointCloud, MyPerspectiveCamera, MyPosture

def fastMVSnet_vis_crop():
    imgs = []
    deps = []
    cams = []

    for ref_idx in [0, 1, 2, 3, 4]:
        # for flow in ['init']:
        for flow in ['flow3']:
            img, depth, intr, extr = {}, {}, {}, {}
            depth_tmp = load_pfm('/data/FastMVSNet/lab/lab_crop/scan1/0000000{}_{}.pfm'.format(ref_idx, flow))
            depth_tmp = depth_tmp[0]
            # depth_tmp = cv2.resize(depth_tmp, [W, H])
            # depth_tmp[depth_tmp < 1.25] = 0
            depth[ref_idx] = depth_tmp
            H, W = depth_tmp.shape
            rgb_tmp = cv2.imread('/data/FastMVSNet/lab/lab_crop/scan1/0000000{}.jpg'.format(ref_idx))
            rgb_tmp = cv2.resize(rgb_tmp, [W, H])
            rgb_tmp = cv2.cvtColor(rgb_tmp, cv2.COLOR_RGB2BGR)
            img[ref_idx] = rgb_tmp

            cam = load_cam_dtu(open('/data/FastMVSNet/lab/lab_crop/scan1/cam_0000000{}_{}.txt'.format(ref_idx, flow)))
            extrinsic = cam[0:4][0:4][0]
            intrinsic = cam[0:4][0:4][1]
            intrinsic = intrinsic[0:3, 0:3]
            intr[ref_idx] = intrinsic
            extr[ref_idx] = extrinsic

            imgs.append(img[ref_idx])
            deps.append(depth_tmp)
            # cams.append(
            #     MyPerspectiveCamera(intrinsic, extrinsic, H, W)
            # )
            cam = MyPerspectiveCamera(intrinsic, extrinsic, H, W)

            # pcd_this_frame = vis_pcd_gpu(img, depth, intr, extr, 'cuda:0', vis=False)
            # pcd_this_frame = pcd_this_frame.to_legacy()
            # o3d.visualization.draw_geometries([pcd_this_frame])

            img[ref_idx] = torch.from_numpy(img[ref_idx])
            depth_tmp = torch.from_numpy(depth_tmp.copy())
            intrinsic = torch.from_numpy(intrinsic)
            extrinsic = torch.from_numpy(extrinsic)

            poi, col = generatePointCloud(img[ref_idx], depth_tmp, cam)
            savePointCloud(os.path.join('outputs', 'lab-pc', 'cam{}.ply'.format(ref_idx)), [poi], [col])

if __name__ == '__main__':
    fastMVSnet_vis_crop()