from isb_filter import ISB_Filter
from utils import rgb2yCbCr
import torch
import numpy as np
import cv2
from g3d_utils import *

def vis(img, title = 'vis', cuda = False, norm = False):
    if cuda:
        img = img.cpu().numpy()
    # img = np.log(img)
    #img = 1.0 / img
    if norm:
        maxs = np.max(img)
        img[img < 0] = 0.5
        mins = 0.5#np.min(img)
        img = 255.0 * (img - mins) / (maxs - mins)
        img = img.astype('uint8')
        #img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
    cv2.imshow(title, img)
    cv2.waitKey()

def vis_pcd(img, depth, intr, extr, idx):
    img_dict, depth_dict, intr_dict, extr_dict = {}, {}, {}, {}
    # depth_tmp = cv2.resize(depth_tmp, [W, H])
    # depth_tmp[depth_tmp < 1.25] = 0
    depth_dict[idx] = depth
    H, W = depth.shape
    rgb_tmp = img
    rgb_tmp = cv2.resize(rgb_tmp, [W, H])
    img_dict[idx] = rgb_tmp

    intr_dict[idx] = intr
    extr_dict[idx] = extr

    pcd_this_frame = vis_pcd_gpu(img_dict, depth_dict, intr_dict, extr_dict, 'cuda:0', vis=False)
    pcd_this_frame = pcd_this_frame.to_legacy()
    o3d.visualization.draw_geometries([pcd_this_frame])

def warping_propagation(rgbd_raw = 'rgbd/rgbd_lab.npy', rgbd_checked = 'rgbd/rgbd_lab_warped.npy', output = 'rgbd/rgbd_lab_propagated.npy'):
    sigma_i = 20
    sigma_s = 25

    img = None
    depth = None
    device = 'cuda:0'
    ans = np.load(rgbd_raw, allow_pickle=True).item()
    imgs = ans['rgb']
    ans = np.load(rgbd_checked, allow_pickle=True).item()
    depths = ans['depth']
    masks = ans['mask']
    intrs = ans['intr']
    extrs = ans['extr']

    for idx in [0, 1, 2, 3, 4]:
        H, W, C = imgs[idx].shape
        guide = rgb2yCbCr(torch.tensor(imgs[idx], dtype=torch.float32).to(device)).type(torch.uint8)
        vis(imgs[idx].copy(), title='rgb', norm=False, cuda=False)

        distance_filter = ISB_Filter(1, [W, H], 'cuda:0', '')

        distance_map = depths[idx]
        mask_map = masks[idx]
        #vis(distance_map, title='depth', norm=True, cuda=False)
        distance_map[distance_map > 1.6] = 1.6
        distance_map[distance_map < 1.0] = 1.0
        # vis(distance_map.copy(), title='depth', norm=True, cuda=False)

        # mask_random = np.random.rand(distance_map.shape[0], distance_map.shape[1]).astype(np.f loat32)
        # mask_random[mask_random < 0.3] = -1
        # mask_random[mask_random > 0] = 1
        # distance_map = distance_map * mask_random

        mask_map = mask_map.astype(np.float32)
        distance_map[mask_map < 0.9] *= -1.0
        vis(distance_map.copy(), title='After Consistency Check', norm=True, cuda=False)
        vis_pcd(imgs[idx], distance_map, intrs[idx].numpy(), extrs[idx].numpy(), idx)

        distance_map = np.expand_dims(distance_map, axis=0)
        distance_map = torch.from_numpy(distance_map).to(device)
        filtered_distance, _ = distance_filter.apply(guide.clone(), distance_map.clone(), sigma_i / 2,
                                                             sigma_s / 2, 2)
        distance_map = filtered_distance[0]
        ans['depth'][idx] = distance_map.cpu().numpy()
        vis(distance_map.clone(), title='After Propagation', norm=True, cuda=True)

        vis_pcd(imgs[idx], distance_map.cpu().numpy(), intrs[idx].numpy(), extrs[idx].numpy(), idx)

    np.save(output, ans)


def random_propagation():
    sigma_i = 20
    sigma_s = 25

    img = None
    depth = None
    device = 'cuda:0'
    ans = np.load('rgbd/rgbd_by_fastMVSnet.npy', allow_pickle=True).item()
    imgs = ans['rgb']
    depths = ans['depth']

    idx = 1
    H, W, C = imgs[idx].shape
    guide = rgb2yCbCr(torch.tensor(imgs[idx], dtype=torch.float32).to(device)).type(torch.uint8)
    vis(imgs[idx].copy(), title='rgb', norm=False, cuda=False)

    distance_filter = ISB_Filter(1, [W, H], 'cuda:0', '')

    distance_map = depths[idx]
    #vis(distance_map, title='depth', norm=True, cuda=False)
    distance_map[distance_map > 1.6] = 1.6
    distance_map[distance_map < 1.0] = 1.0
    vis(distance_map.copy(), title='depth', norm=True, cuda=False)

    mask_random = np.random.rand(distance_map.shape[0], distance_map.shape[1]).astype(np.float32)
    mask_random[mask_random < 0.9] = -1
    mask_random[mask_random > 0] = 1
    distance_map = distance_map * mask_random
    vis(distance_map.copy(), title='mask_random', norm=True, cuda=False)

    distance_map = np.expand_dims(distance_map, axis=0)
    distance_map = torch.from_numpy(distance_map).to(device)
    filtered_distance, _ = distance_filter.apply(guide.clone(), distance_map.clone(), sigma_i / 2,
                                                         sigma_s / 2, 2)
    distance_map = filtered_distance[0]
    vis(distance_map.clone(), title='filtered', norm=True, cuda=True)

#warping_propagation()
#random_propagation()