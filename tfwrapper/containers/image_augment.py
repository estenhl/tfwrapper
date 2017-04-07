import numpy as np
from tfwrapper import twimage
from tfwrapper.containers.image_dataset import ImageContainer


RESIZE = "resize"
TRANSFORM_BW = "bw"
FLIP_LR = "fliplr"
FLIP_UD = "flipud"


def create_name(name, suffixes):
    img_part = name.rsplit(".", 1)

    suffix_string = "_".join(suffixes)

    return "{}_{}.{}".format(img_part[0], suffix_string, img_part[1])


class ImagePreprocess():
    def __init__(self):
        self.augs = {}

    # (width, heiht)
    def resize(self, img_size=(299, 299)):
        self.augs[RESIZE] = img_size

    def bw(self):
        self.augs[TRANSFORM_BW] = True

    def append_flip_lr(self):
        self.augs[FLIP_LR] = True

    def append_flip_ud(self):
        self.augs[FLIP_UD] = True

    def apply_dataset(self, dataset: ImageContainer):
        names, image_paths, labels = dataset.get_data()

        processed_names = []
        processed_imgs = []
        processed_labels = []

        for name, path, label in zip(names, image_paths, labels):
            p_names, imgs = self.apply_file(path, name)
            for p_name, img in zip(p_names, imgs):
                processed_names.append(p_name)
                processed_imgs.append(img)
                processed_labels.append(label)

        return processed_names, processed_imgs, processed_labels

    def apply_file(self, image_path, name):
        img = twimage.imread(image_path)
        return self.apply(img, name)

    def apply(self, img, name):
        img_versions = []
        img_names = []

        org_suffixes = []

        if RESIZE in self.augs:
            img = twimage.resize(img, self.augs[RESIZE])
            #Should check for size
            org_suffixes.append(RESIZE)
        if TRANSFORM_BW in self.augs:
            img = twimage.bw(img, shape=3)
            org_suffixes.append(TRANSFORM_BW)

        img_versions.append(img)
        img_names.append(create_name(name, org_suffixes))

        # img_versions.append(img)
        # img_names.append(org_suffixes)
        # #Append
        if FLIP_LR in self.augs:
            img_versions.append(np.fliplr(img))
            org_suffixes.append(FLIP_LR)
            img_names.append(create_name(name, org_suffixes))
            org_suffixes.remove(FLIP_LR)

        if FLIP_UD in self.augs:
            img_versions.append(np.flipud(img))
            org_suffixes.append(FLIP_UD)
            img_names.append(create_name(name, org_suffixes))
            org_suffixes.remove(FLIP_UD)
        return img_names, img_versions

#
# class ImageAugment():
#     def __init__(self, seed=None):
#         self.augs = {}
#         self.seed = seed
#
#     def add_blur(self, factor=1):
#         self.augs[TRANSFORM_BW] = 1
#
#     def add_flip_left_right(self, factor=1):
#         self.augs[FLIP_LR] = 1
#
#     def add_flip_up_down(self, factor=1):
#         self.augs[FLIP_UD] = 1
#
#     def random_rotate(self, max_angle, factor=0.5):
#         pass
#
#     def random_flip_left_right(self):
#         pass
#
#     def create_data(self, data: ImageContainer, cache):
#         names, image_paths, labels = data.get_data()
#         for i in range(len(names)):
#             apply_suffix = self.generate_suffixes()
#         pass
#
#     def generate_suffixes(self):
#         apply_suffix = []
#         if TRANSFORM_BW in self.augs:
#             apply_suffix.append(TRANSFORM_BW)
#         if FLIP_LR in self.augs:
#             apply_suffix.append(FLIP_LR)
#         if FLIP_UD in self.augs:
#             apply_suffix.append(FLIP_UD)
#
#         return apply_suffix





