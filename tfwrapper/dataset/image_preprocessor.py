import os
import numpy as np

from tfwrapper import twimage

# TODO (16.05.17): This should be rewritten to match the other preprocessor params
ROTATED = 'rotated'
ROTATION_STEPS = 'rotation_steps'
MAX_ROTATION_ANGLE = 'max_rotation_angle'
BLURRED = 'blurred'
BLUR_STEPS = 'blur_steps'
MAX_BLUR_SIGMA = 'max_blur_sigma'


def create_name(name, suffixes):
    return "_".join([name] + suffixes)


class ImagePreprocessor():
    resize_to = False
    bw = False
    flip_lr = False
    flip_ud = False
    blur = False
    rotate = False

    rotated = False
    rotation_steps = 0
    max_rotation_angle = 0.0

    augs = {}

    def rotate(self, rotation_steps=1, max_rotation_angle=10):
        self.rotated = True
        self.rotation_steps = rotation_steps
        self.augs[ROTATED] = {ROTATION_STEPS: rotation_steps, MAX_ROTATION_ANGLE: max_rotation_angle}
        self.augs[ROTATION_STEPS] = rotation_steps
        self.augs[MAX_ROTATION_ANGLE] = max_rotation_angle

    def blur(self, blur_steps=1, max_blur_sigma=1):
        self.augs[BLURRED] = {BLUR_STEPS: blur_steps, MAX_BLUR_SIGMA: max_blur_sigma}

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

        if ROTATED in self.augs:
            rotation_steps = self.augs[ROTATED][ROTATION_STEPS]
            max_rotation_angle = self.augs[ROTATED][MAX_ROTATION_ANGLE]
            for i in range(rotation_steps):
                angle = max_rotation_angle * (i + 1) / rotation_steps

                org_suffixes.append(ROTATED)

                org_suffixes.append(str(angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(angle))

                org_suffixes.append(str(-angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(-angle))

                org_suffixes.remove(ROTATED)

        if BLURRED in self.augs:
            blur_steps = self.augs[BLURRED][BLUR_STEPS]
            max_blur_sigma = self.augs[BLURRED][MAX_BLUR_SIGMA]
            for i in range(blur_steps):
                sigma = max_blur_sigma * (i + 1) / blur_steps
                org_suffixes.append(BLURRED)
                org_suffixes.append(str(sigma))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(sigma))
                org_suffixes.remove(BLURRED)

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

        if ROTATED in self.augs:
            rotation_steps = self.augs[ROTATED][ROTATION_STEPS]
            max_rotation_angle = self.augs[ROTATED][MAX_ROTATION_ANGLE]
            for i in range(rotation_steps):
                angle = max_rotation_angle * (i + 1) / rotation_steps
                imgs.append(twimage.rotate(img, angle))
                org_suffixes.append(ROTATED)
                org_suffixes.append(str(angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(angle))

                imgs.append(twimage.rotate(img, -angle))
                org_suffixes.append(str(-angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(-angle))
                org_suffixes.remove(ROTATED)

        if BLURRED in self.augs:
            blur_steps = self.augs[BLURRED][BLUR_STEPS]
            max_blur_sigma = self.augs[BLURRED][MAX_BLUR_SIGMA]
            for i in range(blur_steps):
                sigma = max_blur_sigma * (i + 1) / blur_steps
                imgs.append(twimage.blur(img, sigma))
                org_suffixes.append(BLURRED)
                org_suffixes.append(str(sigma))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(sigma))
                org_suffixes.remove(BLURRED)

        # TODO: generate combinations of flip, rotation and blur

        return imgs, names