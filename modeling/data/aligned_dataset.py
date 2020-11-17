import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset


class AlignedDataset(Dataset):
    def __init__(self, root_path, transform, mode="train"):
        super(AlignedDataset, self).__init__()
        self.transform = transform
        self.file_src = sorted(glob.glob(os.path.join(root_path, "{}_A".format(mode), "/*.*")))
        self.file_dst = sorted(glob.glob(os.path.join(root_path, "{}_B".format(mode), "/*.*")))
        # self.file_src, self.file_dst = self.get_pair_images()

    def __getitem__(self, index):
        src_img = Image.open(self.file_src[index])
        dst_img = Image.open(self.file_dst[index])
        src_img = self.transform(src_img)
        dst_img = self.transform(dst_img)
        return {"src": src_img, "dst": dst_img}

    def __len__(self):
        return len(self.file_src)
