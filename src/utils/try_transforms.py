import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorchvideo import transforms as videotransforms

from src.utils.i3d_data import I3Dataset
from src.utils.util import load_gzip
from src.utils.videotransforms import RandomCrop
from src.utils import spatial_transforms

cngt_zip = "D:/Thesis/datasets/cngt_single_signs.zip"
sb_zip = "D:/Thesis/datasets/NGT_Signbank_resized.zip"
sb_vocab_path = "D:/Thesis/datasets/signbank_vocab.gzip"
mode = "rgb"
split = "train"
window_size = 16
num_top_glosses = 2
batch_size = 4
specific_glosses = ["PT-1hand", "PO"]

transforms = transforms.Compose([
            transforms.RandomPerspective(0.5),
            transforms.RandomAffine(degrees=10),
            transforms.RandomHorizontalFlip(),
            spatial_transforms.ColorJitter(num_in_frames=window_size),
            transforms.RandomCrop(244)
        ])

sb_vocab = load_gzip(sb_vocab_path)
gloss_to_id = sb_vocab['gloss_to_id']

specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

dataset = I3Dataset(loading_mode="random",
                            cngt_zip=cngt_zip,
                            sb_zip=sb_zip,
                            sb_vocab_path=sb_vocab_path,
                            mode="rgb",
                            split=split,
                            window_size=window_size,
                            transforms=None,
                            filter_num=None,
                            specific_gloss_ids=specific_gloss_ids,
                            clips_per_class=100)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

images, labels = next(iter(dataloader))
images = images.permute([0, 2, 3, 4, 1])

fig = plt.figure()

for i in range(batch_size):
    ax = fig.add_subplot(1, batch_size, i + 1)
    ax.imshow(images[i][0])

plt.show()
