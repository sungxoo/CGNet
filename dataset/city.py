import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2


class Cityscapes(Dataset):
    mean = [73.1545, 82.90554, 72.3905]  # RGB
    std = [46.996418, 47.91469, 47.137108]

    category = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    palette = np.asarray([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                          [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                          [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                          [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    ignore_idx = 255
    ignore_color = np.asarray([0, 0, 0])
    id_maps = [255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255,
               6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18]
    license_plate_id = -1

    def __init__(self, root_dir, split='train', transform=None):
        '''
        :param root_dir: dataset root path
        :param transform: optional transform
        '''
        super(Cityscapes, self).__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.img_path = os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit')
        self.seg_path = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine')

        self.ids = []
        self.imgs = []
        self.segs = []

        self.imgs.extend([os.path.join(root, filename)
                            for root, _, filenames in os.walk(os.path.join(self.img_path, split))
                            for filename in filenames if filename.endswith('.png')])
        self.segs.extend([os.path.join(root, filename)
                            for root, _, filenames in os.walk(os.path.join(self.seg_path, split))
                            for filename in filenames if filename.endswith('.png')])
        self.ids.extend([os.path.splitext(img.split(os.sep)[-1])[0] for img in self.imgs])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(self.segs[idx])

        seg[seg == self.license_plate_id] = self.ignore_idx
        for i, id enumerate(self.id_maps):
            seg[seg == i] = id

        output = {'image': img, 'label': seg}

        if self._transform is not None:
            output = self._transform(output)

        return output

    def _abnormalize(self, img):
        with torch.no_grad():
            img = img.detach().cpu()
            img = np.array(img).astype(np.float32).transpose(1, 2, 0)
            img = ((img * self.std) + self.mean) * 255.0

        return img.astype(dtype=np.uint8)

    def _seg2rgb(self, seg):
        with torch.no_grad():
            seg = seg.detach().cpu()
            seg = np.array(seg).astype(np.float32)

            r = seg.copy()
            g = seg.copy()
            b = seg.copy()

            rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

            # Class pixels are painted
            for i in range(self.n_category):
                r[seg == i] = self.palette[i, 0]
                g[seg == i] = self.palette[i, 1]
                b[seg == i] = self.palette[i, 2]

            # Ignore pixels are painted
            r[seg == self.ignore_index] = self.ignore_palette[0]
            g[seg == self.ignore_index] = self.ignore_palette[1]
            b[seg == self.ignore_index] = self.ignore_palette[2]

            rgb[:, :, 0] = r
            rgb[:, :, 1] = g
            rgb[:, :, 2] = b

        return rgb

    def make_grid(self, imgs, segs, predictions):
        batch_size = imgs.shape[0]
        grid_imgs = []

        for i in range(batch_size):
            img = self._abnormalize(imgs[i])
            img1 = img.copy()
            seg = self._seg2rgb(segs[i]) # seg.shape :[768, 768, 3]
            seg1 = seg.copy()
            prediction = self._seg2rgb(predictions[i]) # prediction.shape :[768, 768, 3]

            # prediction1 = np.where(seg == [0],[0], prediction)
            prediction1 = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
            prediction1[:, :, 0] = np.where(segs[i].cpu() == self.ignore_index, [0], prediction[:, :, 0])
            prediction1[:, :, 1] = np.where(segs[i].cpu() == self.ignore_index, [0], prediction[:, :, 1])
            prediction1[:, :, 2] = np.where(segs[i].cpu() == self.ignore_index, [0], prediction[:, :, 2])
            prediction1 = np.array(prediction1).astype(np.uint8)
            grid_imgs.append(np.concatenate((img1, seg1, prediction1), axis=1).transpose(2, 0, 1))

        return grid_imgs




if __name__ is '__main__':
    city = Cityscapes()