import os
import numpy as np

from tfwrapper import twimage


def create_name(name, suffixes):
    return "_".join([name] + suffixes)


class ImagePreprocessor():
    resize_to = False
    bw = False
    flip_lr = False
    flip_ud = False
    blur = False
    rotate = False

    rotate = False
    rotation_steps = 0
    max_rotation_angle = 0.0

    blur = False
    blur_steps = 0
    max_blur_sigma = 0.0

    def rotate(self, rotation_steps=1, max_rotation_angle=10):
        self.rotate = True
        self.rotation_steps = rotation_steps
        self.max_rotation_angle = max_rotation_angle

    def blur(self, blur_steps=1, max_blur_sigma=1):
        self.blur = True
        self.blur_steps = blur_steps
        self.max_blur_sigma = max_blur_sigma

    def get_names(self, path, name):
        if name is None:
            name = '.'.join(os.path.basename(path).split('.')[:-1])

        org_suffixes = []
        names = []

        if self.resize_to:
            width, height = self.resize_to
            org_suffixes.append('%s%dx%d' % ('resize', width, height))
        if self.bw:
            org_suffixes.append('bw')

        names.append(create_name(name, org_suffixes))

        if self.flip_lr:
            org_suffixes.append('fliplr')
            names.append(create_name(name, org_suffixes))
            org_suffixes.remove('fliplr')

        if self.flip_ud:
            org_suffixes.append('flipud')
            names.append(create_name(name, org_suffixes))
            org_suffixes.remove('flipud')

        if self.rotate:
            for i in range(self.rotation_steps):
                angle = self.max_rotation_angle * (i + 1) / self.rotation_steps

                org_suffixes.append('rotated')

                org_suffixes.append(str(angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(angle))

                org_suffixes.append(str(-angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(-angle))

                org_suffixes.remove('rotated')

        if self.blur:
            for i in range(self.blur_steps):
                sigma = self.max_blur_sigma * (i + 1) / self.blur_steps
                org_suffixes.append('blurred')
                org_suffixes.append(str(sigma))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(sigma))
                org_suffixes.remove('blurred')

        # TODO: generate combinations of flip, rotation and blur

        return names

    def process(self, img, name, label=None):
        if img is None:
            return [], []

        imgs = []
        names = []

        org_suffixes = []

        if self.resize_to:
            img = twimage.resize(img, self.resize_to)
            # Should check for size
            width, height = self.resize_to
            org_suffixes.append('%s%dx%d' % ('resize', width, height))
        if self.bw:
            img = twimage.bw(img, shape=3)
            org_suffixes.append('bw')

        imgs.append(img)
        names.append(create_name(name, org_suffixes))

        if self.flip_lr:
            imgs.append(np.fliplr(img))
            org_suffixes.append('fliplr')
            names.append(create_name(name, org_suffixes))
            org_suffixes.remove('fliplr')

        if self.flip_ud:
            imgs.append(np.flipud(img))
            org_suffixes.append('flipud')
            names.append(create_name(name, org_suffixes))
            org_suffixes.remove('flipud')

        if self.rotate:
            for i in range(self.rotation_steps):
                angle = self.max_rotation_angle * (i + 1) / self.rotation_steps
                imgs.append(twimage.rotate(img, angle))
                org_suffixes.append('rotated')
                org_suffixes.append(str(angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(angle))

                imgs.append(twimage.rotate(img, -angle))
                org_suffixes.append(str(-angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(-angle))
                org_suffixes.remove('rotated')

        if self.blur:
            for i in range(self.blur_steps):
                sigma = self.max_blur_sigma * (i + 1) / self.blur_steps
                imgs.append(twimage.blur(img, sigma))
                org_suffixes.append('blurred')
                org_suffixes.append(str(sigma))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(sigma))
                org_suffixes.remove('blurred')

        # TODO (22.06.17): generate combinations of flip, rotation and blur

        return imgs, names