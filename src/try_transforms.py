import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorchvideo import transforms as videotransforms

from src.utils.i3d_data import I3Dataset
from src.utils.videotransforms import RandomCrop
from src.utils import spatial_transforms

cngt_zip = "D:/Thesis/datasets/cngt_train_clips.zip"
sb_zip = "D:/Thesis/datasets/NGT_Signbank_resized.zip"
mode = "rgb"
window_size = 16
num_top_glosses = 2
batch_size = 4


transforms = transforms.Compose([
            transforms.RandomPerspective(0.5),
            transforms.RandomAffine(degrees=10),
            transforms.RandomHorizontalFlip(),
            spatial_transforms.ColorJitter(num_in_frames=window_size),
            transforms.RandomCrop(244)
        ])

dataset = I3Dataset(cngt_zip, sb_zip, mode, 'train', window_size, transforms=transforms, filter_num=num_top_glosses)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

images, labels = next(iter(dataloader))
images = images.permute([0, 2, 3, 4, 1])

fig = plt.figure()

for i in range(batch_size):
    ax = fig.add_subplot(1, batch_size, i + 1)
    ax.imshow(images[i][0])

plt.show()
