import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import sys
sys.path.append('/home/wjk/workspace/PyProject/FastMVSNet/fastmvsnet')
from picutils import PICTimer

class FeatureFetcher(nn.Module):
    def __init__(self, mode="bilinear"):
        super(FeatureFetcher, self).__init__()
        self.mode = mode

    def forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
            grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

        # pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode, padding_mode='border')
        # print("without border pad-----------------------")
        pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode)
        pts_feature = pts_feature.squeeze(3)

        pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)

        return pts_feature


class FeatureGradFetcher(nn.Module):
    def __init__(self, mode="bilinear"):
        super(FeatureGradFetcher, self).__init__()
        self.mode = mode

    def grid_sample(self, image, optical):
        N, C, IH, IW = image.shape
        _, H, W, _ = optical.shape

        ix = optical[..., 0]
        iy = optical[..., 1]

        ix = ((ix + 1) / 2) * (IW - 1);
        iy = ((iy + 1) / 2) * (IH - 1);
        with torch.no_grad():
            ix_nw = torch.floor(ix);
            iy_nw = torch.floor(iy);
            ix_ne = ix_nw + 1;
            iy_ne = iy_nw;
            ix_sw = ix_nw;
            iy_sw = iy_nw + 1;
            ix_se = ix_nw + 1;
            iy_se = iy_nw + 1;

        nw = (ix_se - ix) * (iy_se - iy)
        ne = (ix - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix) * (iy - iy_ne)
        se = (ix - ix_nw) * (iy - iy_nw)

        with torch.no_grad():
            torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
            torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

            torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
            torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

            torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
            torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

            torch.clamp(ix_se, 0, IW - 1, out=ix_se)
            torch.clamp(iy_se, 0, IH - 1, out=iy_se)

        image = image.view(N, C, IH * IW)

        nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
        ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
        sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
        se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

        out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
                   ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
                   sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
                   se_val.view(N, C, H, W) * se.view(N, 1, H, W))

        return out_val

    def forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
            grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

            #todo check bug
            grid_l = grid.clone()
            grid_l[..., 0] -= (1. / float(width - 1)) * 2

            grid_r = grid.clone()
            grid_r[..., 0] += (1. / float(width - 1)) * 2

            grid_t = grid.clone()
            grid_t[..., 1] -= (1. / float(height - 1)) * 2

            grid_b = grid.clone()
            grid_b[..., 1] += (1. / float(height - 1)) * 2


        def get_features(grid_uv):
            pts_feature = F.grid_sample(feature_maps, grid_uv, mode=self.mode)
            pts_feature = pts_feature.squeeze(3)

            pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)
            return pts_feature

        pts_feature = get_features(grid)

        pts_feature_l = get_features(grid_l)
        pts_feature_r = get_features(grid_r)
        pts_feature_t = get_features(grid_t)
        pts_feature_b = get_features(grid_b)

        pts_feature_grad_x = 0.5 * (pts_feature_r - pts_feature_l)
        pts_feature_grad_y = 0.5 * (pts_feature_b - pts_feature_t)

        pts_feature_grad = torch.stack((pts_feature_grad_x, pts_feature_grad_y), dim=-1)
        # print("================features++++++++++++")
        # print(feature_maps)
        # print ("===========grad+++++++++++++++")
        # print (pts_feature_grad)
        return pts_feature, pts_feature_grad

    def get_result(self,  feature_maps, pts, cam_intrinsics, cam_extrinsics):
        torch.cuda.synchronize()
        timer_warping = PICTimer.getTimer('warping@d_uv')
        timer_warping.startTimer()
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        num_pts = pts.size(2)
        pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
            .contiguous().view(curr_batch_size, 3, num_pts)
        if cam_extrinsics is None:
            transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
        else:
            cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
            R = torch.narrow(cam_extrinsics, 2, 0, 3)
            t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
            transformed_pts = torch.bmm(R, pts_expand) + t
            transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
        x = transformed_pts[..., 0]
        y = transformed_pts[..., 1]
        z = transformed_pts[..., 2]

        normal_uv = torch.cat(
            [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
            dim=-1)
        uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
        uv = uv[:, :, :2]

        grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
        grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
        grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

        def get_features(grid_uv):
            # pts_feature = F.grid_sample(feature_maps, grid_uv, mode=self.mode)
            pts_feature = self.grid_sample(feature_maps, grid_uv)
            pts_feature = pts_feature.squeeze(3)

            pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)
            # return pts_feature.detach()
            return pts_feature

        pts_feature = get_features(grid)
        torch.cuda.synchronize()
        timer_warping.showTime('warping')

        grid = grid.detach().clone()

        # todo check bug
        grid[..., 0] -= (1. / float(width - 1)) * 2
        pts_feature_l = get_features(grid)
        grid[..., 0] += (1. / float(width - 1)) * 2

        grid[..., 0] += (1. / float(width - 1)) * 2
        pts_feature_r = get_features(grid)
        grid[..., 0] -= (1. / float(width - 1)) * 2

        grid[..., 1] -= (1. / float(height - 1)) * 2
        pts_feature_t = get_features(grid)
        grid[..., 1] += (1. / float(height - 1)) * 2

        grid[..., 1] += (1. / float(height - 1)) * 2
        pts_feature_b = get_features(grid)
        grid[..., 1] -= (1. / float(height - 1)) * 2

        pts_feature_r -= pts_feature_l
        pts_feature_r *= 0.5
        pts_feature_b -= pts_feature_t
        pts_feature_b *= 0.5
        torch.cuda.synchronize()
        timer_warping.showTime('d_uv')
        # timer_warping.summary()
        # return pts_feature.detach(), pts_feature_r.detach(), pts_feature_b.detach()
        return pts_feature, pts_feature_r.detach(), pts_feature_b.detach()

    def test_forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        # with torch.no_grad():
        with torch.set_grad_enabled(True):
            pts_feature, grad_x, grad_y = \
                self.get_result(feature_maps, pts, cam_intrinsics, cam_extrinsics)
        # torch.cuda.synchronize()
        # begin_t = time.time()
        torch.cuda.empty_cache()
        pts_feature_grad = torch.stack((grad_x, grad_y), dim=-1)
        # torch.cuda.synchronize()
        # print('    stack: ', time.time() - begin_t)
        return pts_feature, pts_feature_grad.detach()


class PointGrad(nn.Module):
    def __init__(self):
        super(PointGrad, self).__init__()

    def forward(self, pts, cam_intrinsics, cam_extrinsics):
        """
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, _, _ = list(cam_extrinsics.size())

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        # with torch.no_grad():
        with torch.set_grad_enabled(True):
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            fx = cam_intrinsics[..., 0, 0].view(curr_batch_size, 1)
            fy = cam_intrinsics[..., 1, 1].view(curr_batch_size, 1)

            # print("x", x.size())
            # print("fx", fx.size(), fx, fy)

            zero = torch.zeros_like(x)
            grad_u = torch.stack([fx / z, zero, -fx * x / (z**2)], dim=-1)
            grad_v = torch.stack([zero, fy / z, -fy * y / (z**2)], dim=-1)
            grad_p = torch.stack((grad_u, grad_v), dim=-2)
            # print("grad_u size:", grad_u.size())
            # print("grad_p size:", grad_p.size())
            grad_p = grad_p.view(batch_size, num_view, num_pts, 2, 3)
        return grad_p



class ProjectUVFetcher(nn.Module):
    def __init__(self, mode="bilinear"):
        super(ProjectUVFetcher, self).__init__()
        self.mode = mode

    def forward(self, pts, cam_intrinsics, cam_extrinsics):
        """

        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view = cam_extrinsics.size()[:2]

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)

        return grid.view(batch_size, num_view, num_pts, 1, 2)


def test_feature_fetching():
    import numpy as np
    batch_size = 3
    num_view = 2
    channels = 16
    height = 240
    width = 320
    num_pts = 32

    cam_intrinsic = torch.tensor([[10, 0, 1], [0, 10, 1], [0, 0, 1]]).float() \
        .view(1, 1, 3, 3).expand(batch_size, num_view, 3, 3).cuda()
    cam_extrinsic = torch.rand(batch_size, num_view, 3, 4).cuda()

    feature_fetcher = FeatureFetcher().cuda()

    features = torch.rand(batch_size, num_view, channels, height, width).cuda()

    imgpt = torch.tensor([60.5, 80.5, 1.0]).view(1, 1, 3, 1).expand(batch_size, num_view, 3, num_pts).cuda()

    z = 200

    pt = torch.matmul(torch.inverse(cam_intrinsic), imgpt) * z

    pt = torch.matmul(torch.inverse(cam_extrinsic[:, :, :, :3]),
                      (pt - cam_extrinsic[:, :, :, 3].unsqueeze(-1)))  # Xc = [R|T] Xw

    gathered_feature = feature_fetcher(features, pt[:, 0, :, :], cam_intrinsic, cam_extrinsic)

    gathered_feature = gathered_feature[:, 0, :, 0]
    np.savetxt("gathered_feature.txt", gathered_feature.detach().cpu().numpy(), fmt="%.4f")

    groundtruth_feature = features[:, :, :, 80, 60][:, 0, :]
    np.savetxt("groundtruth_feature.txt", groundtruth_feature.detach().cpu().numpy(), fmt="%.4f")

    print(np.allclose(gathered_feature.detach().cpu().numpy(), groundtruth_feature.detach().cpu().numpy(), 1.e-2))


if __name__ == "__main__":
    test_feature_fetching()
