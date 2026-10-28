from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageDataset, self).__init__()
        self.root = root
        self.transform = transform  
        self.files=list(os.listdir(root))
        self.files=[p for p in self.files if p.endswith('.png', '.jpg', '.jpeg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)