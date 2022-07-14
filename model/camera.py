import torch

def get_camera_params(opt):
    """
    decide camera intrinsics from configurations
    """
    if opt is None:
        print("Warning: using default kinect camera intrinsics with crop size 1200")
        fx = 979.7844 / 2048
        fy = 979.7844 / 2048
        cx = 1018.952 / 2048.
        cy = 779.486 / 2048.
        crop_size = 1200
    else:
        fx = 979.7844 / 2048 if 'fx' not in opt else opt.fx
        fy = 979.7844 / 2048 if 'fy' not in opt else opt.fy
        cx = 1018.952 / 2048. if 'cx' not in opt else opt.cx
        cy = 779.486 / 2048. if 'cx' not in opt else opt.cy
        crop_size = opt.loadSize
    print(f'using camera parameters: crop size={crop_size}, fx={fx}, fy={fy}, cx={cx}, cy={cy}')
    return crop_size, cx, cy, fx, fy


class KinectColorCamera:
    "kinect color camera config"
    def __init__(self, crop_size=1200, fx=979.7844/2048., fy=979.840/2048.,
                 cx=1018.952/2048., cy=779.486/2048., image_size=2048):
        # self.fx, self.fy = 979.7844, 979.840
        # self.cx, self.cy = 1018.952, 779.486 # kinect 1 parameters
        # self.width, self.height = 2048, 1536

        self.fx, self.fy = fx, fy # now the normalized focal
        self.cx, self.cy = cx, cy  # normalized camera center
        self.width, self.height = image_size, int(image_size*0.75) # 4x3 image

        # precompute the focal in pixel space
        self.fx_px, self.fy_px = self.fx*image_size, self.fy * image_size
        self.cx_px, self.cy_px = self.cx*image_size, self.cy * image_size

        self.crop_size = crop_size # for cropping setting
        # print('cropping size: {}'.format(crop_size))
        pass

    def project_points(self, points, offset=None):
        "return in normalized"
        px, py = self.project_screen(points) # project to original kinect pixel space
        nx, ny = self.normalize(px, py, offset) # normalize to local patch
        xyzn = torch.cat([nx, ny, points[:, :, 2:3]], -1).transpose(1, 2)
        return xyzn

    def project_screen(self, points, crop_center=None):
        " project points to screen, points: (B, N, 3)"
        # Update March 4: use normalized focal length
        if len(points.shape) == 3:
            # batched points for trianing
            x, y, z = points[:, :, 0:1], points[:, :, 1:2], points[:, :, 2:3]
        elif len(points.shape) == 2:
            # non-batch
            x, y, z = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        else:
            raise NotImplemented
        # px = (self.fx * x / z + self.cx)*self.width
        # py = (self.fy * y / z + self.cy)*self.width
        px = self.fx_px * x / z + self.cx_px
        py = self.fy_px * y / z + self.cy_px
        if crop_center is not None:
            # take crop into account
            px = self.crop_size / 2 + px - crop_center[:, 0].unsqueeze(1).unsqueeze(1)
            py = self.crop_size / 2 + py - crop_center[:, 1].unsqueeze(1).unsqueeze(1)  # align with cropped center
        return px, py

    def normalize(self, px, py, offset=None):
        "normalize to (-1, 1), offset: (B, 2), px, py: (B, N, 1)"
        if offset is not None:
            px = self.crop_size/2 + px - offset[:, 0].unsqueeze(1).unsqueeze(1)
            py = self.crop_size / 2 + py - offset[:, 1].unsqueeze(1).unsqueeze(1) # align with cropped center
            nx = 2*px/self.crop_size - 1
            ny = 2*py/self.crop_size - 1
            # print('crop center: {}'.format(offset[0]))
            assert px.shape[-1] == 1,'invalid shape of px found: {}'.format(px.shape)
            assert py.shape[-1] == 1, 'invalid shape found: {}'.format(py.shape)
            # print("cropping size: {}".format(self.crop_size))
        else:
            assert self.fx > 1.0, 'error, trying to project using incompatible intrinsics'
            nx = 2*px/self.width - 1
            ny = 2*py/self.height - 1
            # print("No cropping")
        return nx, ny


class KinectOrthCamera:
    "approximate orthographic camera"
    def __init__(self, load_size=512, scale=0.75):
        self.load_size = 512
        self.scale = scale # scale for the person so the object can be included

    def project_points(self, points, offset=None):
        "assume points are move to the smpl center, points: (B, N, 3)"
        return points.transpose(1, 2)
