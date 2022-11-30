import zipfile
import os

import torchvision.transforms as transforms
import numpy as np

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # (384, 384) if vision transformer
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

from torch.utils.data import Dataset
from PIL import Image

# for self supervised learning
class ssl_dataset(Dataset):
    def __init__(self):
        root = "D://nabirds/images"
        self.image_paths = []
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            for im in os.listdir(path):
                self.image_paths.append(os.path.join(path, im))

        # demo
        # self.image_paths = self.image_paths[:100]
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path)
        image = transforms.Resize((128, 128))(image)
        gray_image = transforms.Grayscale(num_output_channels=3)(image)
        image = np.array(image)/255
        gray_image = np.array(gray_image)/255
        if image.shape[-1] == 4: # RGBA
            image = image[:,:,:3]
        return gray_image.transpose(2,0,1).astype(np.float32), image.transpose(2,0,1).astype(np.float32)


def show(imgs):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    import numpy as np

    if not isinstance(imgs, list) and not len(imgs.shape) >= 4:
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        # img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.show()


if __name__ == "__main__":
    pass
# Data Augmentation
#     import os
#     import PIL
#     from PIL import Image, ImageEnhance, ImageFilter
#     import numpy as np
#     from tqdm import tqdm

    # mainpath = "./bird_dataset/train_images"
    # def rotate(image, save_path, imagename, angles = [90, 180, 270]):
    #     for angle in angles:
    #         new_image = image.rotate(angle, expand = True)
    #         new_image.save(os.path.join(save_path, f"rotate_{angle}_" + imagename))
    # def add_noise(image, save_path, imagename, sigma = 10):
    #     new_image = np.copy(np.array(image))/255
    #     noise = sigma * np.random.normal(0, 1, new_image.shape)/255
    #     new_image = 255 * np.clip(new_image + noise, 0, 1)
    #     new_image = new_image.astype(np.uint8)
    #     new_image = Image.fromarray(new_image)
    #     new_image.save(os.path.join(save_path, f"noise_{sigma}_" + imagename))
    # def add_contrast(image, save_path, imagename, factors = [1.1, 1.5]):
    #     enhancer = ImageEnhance.Contrast(image)
    #     for factor in factors:
    #         new_image = enhancer.enhance(factor)
    #         new_image.save(os.path.join(save_path, f"contrast_{factor}_" + imagename))
    # def add_brightness(image, save_path, imagename, factors = [1.2, 0.7]):
    #     enhancer = ImageEnhance.Brightness(image)
    #     for factor in factors:
    #         new_image = enhancer.enhance(factor)
    #         new_image.save(os.path.join(save_path, f"brightness_{factor}_" + imagename))
    # def Gaussianblur(image, save_path, imagename, radius = 2):
    #     Gaussianfilter = ImageFilter.GaussianBlur(radius = radius)
    #     new_image = image.filter(Gaussianfilter)
    #     new_image.save(os.path.join(save_path, f"Gaussianblur_{radius}" + imagename))
    #
    # for filename in tqdm(os.listdir(mainpath)):
    #     path = os.path.join(mainpath, filename)
    #     for im in os.listdir(path):
    #
    #         save_path = os.path.join("./bird_dataset/noise", filename)
    #         if not os.path.isdir(save_path):
    #             os.mkdir(save_path)
    #         image_path = os.path.join(path, im)
    #         image = Image.open(image_path)
    #         rotate(image, save_path, im)
    #         add_noise(image, save_path, im)
    #         add_contrast(image, save_path, im)
    #         add_brightness(image, save_path, im)
    #         Gaussianblur(image, save_path, im)

    # Data augmentation
    # from cv2 import cv2
    # import numpy as np
    # import os
    # from PIL import Image
    # from tqdm import tqdm
    #

    # def getDark(image, percentage=0.8):
    #     image_copy = image.copy()
    #     w = image.shape[1]
    #     h = image.shape[0]
    #     image_copy = image_copy * percentage
    #     return image_copy.astype(np.uint8)
    #
    #
    # def getBright(image, percentage=1.5):
    #     image_copy = image.copy()
    #     w = image.shape[1]
    #     h = image.shape[0]
    #     image_copy = np.clip(image_copy * percentage, a_max=255, a_min=0)
    #     return image_copy.astype(np.uint8)
    #
    #
    # def rotate(image, angle, center=None, scale=1.0):
    #     (h, w) = image.shape[:2]
    #     if center is None:
    #         center = (w / 2, h / 2)
    #     m = cv2.getRotationMatrix2D(center, angle, scale)
    #     rotated = cv2.warpAffine(image, m, (w, h))
    #     return rotated.astype(np.uint8)
    #
    #
    # def flip(image):
    #     flipped_image = np.fliplr(image)
    #     return flipped_image.astype(np.uint8)
    #
    # for filename in tqdm(os.listdir(mainpath)):
    #     path = os.path.join(mainpath, filename)
    #     for im in os.listdir(path):
    #         save_path = path
    #         image_path = os.path.join(path, im)
    #         image = Image.open(image_path)
    #         image = np.array(image, dtype=np.float32)
    #
    #         new_image = Image.fromarray(rotate(image, 90))
    #         new_image.save(os.path.join(path, "rotate90" + im))
    #
    #         new_image = Image.fromarray(rotate(image, 270))
    #         new_image.save(os.path.join(path, "rotate270" + im))
    #
    #         new_image = Image.fromarray(flip(image))
    #         new_image.save(os.path.join(path, "flip" + im))
    #
    #         new_image = Image.fromarray(getDark(image))
    #         new_image.save(os.path.join(path, "dark" + im))
    #
    #         new_image = Image.fromarray(getBright(image))
    #         new_image.save(os.path.join(path, "bright" + im))






