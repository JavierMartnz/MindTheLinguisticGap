import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import torch.nn.functional as nnf



try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            if hasattr(t, 'randomize_parameters'):
                t.randomize_parameters()


class ToNumpy(object):
    def __call__(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        elif type(tensor).__module__ != "numpy":
            raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
        return tensor
    def randomize_parameters(self):
        pass

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            # img = torch.from_numpy(pic.transpose((2, 0, 1)))
            img = torch.from_numpy(pic)
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            if isinstance(img, np.ndarray):
                _, w, h = img.shape
            else:
                w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
            if isinstance(img, np.ndarray):
                return np.transpose(cv2.resize(np.transpose(img, (1, 2, 0)), (ow, oh)), (2, 0, 1))
            else:
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
        PIL.Image: Cropped image.
        """
        if isinstance(img, np.ndarray):
            c, h, w = img.shape
            th, tw = self.size
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return img[:, y1: y1 + th, x1: x1 + tw]

        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """

        assert imgs.dim() in {4}, "only 4 dim CTHW tensors are supported"
        supported_types = (torch.FloatTensor, torch.cuda.FloatTensor)
        assert isinstance(imgs, supported_types), "expected single precision inputs"

        if self.p < 0.5:
           return torch.flip(imgs, dims=[-1])
        else:
            return imgs

    def randomize_parameters(self):
        self.p = random.random()

def resize_generic(img, oheight, owidth, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
          |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    # resized_image = cv2.resize(image, (100, 50))
    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        resized_img = scipy.misc.imresize(
            img.squeeze(), [oheight, owidth], interp=interp, mode="F"
        ).reshape((oheight, owidth, chn))
    elif chn == 3:
        # resized_img = scipy.misc.imresize(img, [oheight, owidth], interp=interp)  # mode='F' gives an error for 3 channels
        resized_img = cv2.resize(img, (owidth, oheight))  # inverted compared to scipy
    elif chn == 2:
        # assert(is_flow)
        resized_img = np.zeros((oheight, owidth, chn), dtype=img.dtype)
        for t in range(chn):
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp)
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp, mode='F')
            # resized_img[:, :, t] = np.array(Image.fromarray(img[:, :, t]).resize([oheight, owidth]))
            resized_img[:, :, t] = scipy.ndimage.interpolation.zoom(
                img[:, :, t], [oheight, owidth]
            )
    else:
        in_chn = 3
        # Workaround, would be better to pass #frames
        if chn == 16:
            in_chn = 1
        if chn == 32:
            in_chn = 2
        nframes = int(chn / in_chn)
        img = img.reshape(img.shape[0], img.shape[1], in_chn, nframes)
        resized_img = np.zeros((oheight, owidth, in_chn, nframes), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, :, t]  # img[:, :, t*3:t*3+3]
            frame = cv2.resize(frame, (owidth, oheight)).reshape(
                oheight, owidth, in_chn
            )
            # frame = scipy.misc.imresize(frame, [oheight, owidth], interp=interp)
            resized_img[:, :, :, t] = frame
        resized_img = resized_img.reshape(
            resized_img.shape[0], resized_img.shape[1], chn
        )

    if is_flow:
        # print(oheight / ht)
        # print(owidth / wd)
        resized_img = resized_img * oheight / ht
    return resized_img

class RandomScale(object):
    # tj : [1 - self.scale_factor, 1 + self.scale_factor)
    def __init__(self, resol, scale_factor):
        self.resol = resol
        self.scale_factor = scale_factor

    def __call__(self, imgs):
        resol = self.resol * (1 - self.scale_factor + 2 * self.scale_factor * self.rand_scale)
        resol = int(resol)


        iH, iW = imgs.shape[-2:]
        if iW > iH:
            nH, nW = resol, int(resol * iW / iH)
        else:
            nH, nW = int(resol * iH / iW), resol
            # Resize to nH, nW resolution

        imgs = nnf.interpolate(imgs, size=(nH, nW), mode='bilinear', align_corners=False)
        return imgs

    def randomize_parameters(self):
        self.rand_scale = random.random()


class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w
    def __call__(self, imgs):
        imgs = nnf.interpolate(imgs, size=(self.h, self.w), mode='bilinear')
        return imgs

    def randomize_parameters(self):
        pass






class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
	A crop of size is selected from scales of the original size.
	A position of cropping is randomly selected from 4 corners and 1 center.
	This crop is finally resized to given size.
	Args:
	    scales: cropping scales of the original size
	    size: size of the smaller edge
	    interpolation: Default: PIL.Image.BILINEAR
	"""

    def __init__(self,
                 scales,
                 size,
                 interpolation=Image.BILINEAR,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            min_length = min(img.shape[1], img.shape[2])
            crop_size = int(min_length * self.scale)

            image_width = img.shape[2]
            image_height = img.shape[1]
        else:
            min_length = min(img.size[0], img.size[1])
            crop_size = int(min_length * self.scale)

            image_width = img.size[0]
            image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        if isinstance(img, np.ndarray):
            img = img[:, y1:y2, x1:x2]
            return np.transpose(cv2.resize(np.transpose(img, (1, 2, 0)), (self.size, self.size)), (2, 0, 1))
        else:
            img = img.crop((x1, y1, x2, y2))
            return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(
            0,
            len(self.crop_positions) - 1)]


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()



def im_to_video(img):
    assert img.dim() == 3
    nframes = int(img.size(0) / 3)
    return img.contiguous().view(3, nframes, img.size(1), img.size(2))


def video_to_im(video):
    assert video.dim() == 4
    assert video.size(0) == 3
    return video.view(3 * video.size(1), video.size(2), video.size(3))


class ColorJitter(object):   # tj : remake from bsl1k
    def __init__(self, num_in_frames=1, thr=0.2, deterministic_jitter_val=None):
        self.num_in_frames = num_in_frames
        self.thr = thr
        self.deterministic_jitter_val = deterministic_jitter_val

    def __call__(self, rgb):
        assert rgb.dim() in {3, 4}, "only 3 or 4 dim tensors are supported"
        supported_types = (torch.FloatTensor, torch.cuda.FloatTensor)
        assert isinstance(rgb, supported_types), "expected single precision inputs"
        if rgb.min() < 0:
            print(f"Warning: rgb.min() {rgb.min()} is less than 0.")
        if rgb.max() > 1:
            print(f"Warning: rgb.max() {rgb.max()} is more than 1.")
        if self.deterministic_jitter_val:
            assert (
                    len(self.deterministic_jitter_val) == 3
            ), "expected to be provided 3 fixed vals"
            rjitter, gjitter, bjitter = self.deterministic_jitter_val
        else:
            rjitter = random.uniform(1 - self.thr, 1 + self.thr)
            gjitter = random.uniform(1 - self.thr, 1 + self.thr)
            bjitter = random.uniform(1 - self.thr, 1 + self.thr)
        if rgb.dim() == 3:
            rgb = im_to_video(rgb)
            assert (
                    rgb.shape[1] == self.num_in_frames
            ), "Unexpected number of input frames per clip"
            rgb[0, :, :, :].mul_(rjitter).clamp_(0, 1)
            rgb[1, :, :, :].mul_(gjitter).clamp_(0, 1)
            rgb[2, :, :, :].mul_(bjitter).clamp_(0, 1)
            rgb = video_to_im(rgb)
        elif rgb.dim() == 4:
            assert rgb.shape[0] == 3, "expecte RGB to lie on first axis"
            assert (
                    rgb.shape[1] == self.num_in_frames
            ), "Unexpected number of input frames per clip"
            rgb[0].mul_(rjitter).clamp_(0, 1)
            rgb[1].mul_(gjitter).clamp_(0, 1)
            rgb[2].mul_(bjitter).clamp_(0, 1)
        return rgb
    def randomize_parameters(self):
        pass