import os
import json
import sys
import time

import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from aegan import AEGAN


BATCH_SIZE = 32
LATENT_DIM = 16
EPOCHS = 20000

def save_images(GAN, vec, filename):
    images = GAN.generate_samples(vec)
    ims = tv.utils.make_grid(images[:36], normalize=True, nrow=6,)
    ims = ims.numpy().transpose((1,2,0))
    ims = np.array(ims*255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)


def main():
    os.makedirs("results/generated", exist_ok=True)
    os.makedirs("results/reconstructed", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    root = os.path.join("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fillcolor=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=root,
            transform=transform
            )
    dataloader = DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last=True
            )
    X = iter(dataloader)
    test_ims1, _ = next(X)
    test_ims2, _ = next(X)
    test_ims = torch.cat((test_ims1, test_ims2), 0)
    test_ims_show = tv.utils.make_grid(test_ims[:36], normalize=True, nrow=6,)
    test_ims_show = test_ims_show.numpy().transpose((1,2,0))
    test_ims_show = np.array(test_ims_show*255, dtype=np.uint8)
    image = Image.fromarray(test_ims_show)
    image.save("results/reconstructed/test_images.png")

    noise_fn = lambda x: torch.randn((x, LATENT_DIM), device=device)
    test_noise = noise_fn(36)
    gan = AEGAN(
        LATENT_DIM,
        noise_fn,
        dataloader,
        device=device,
        batch_size=BATCH_SIZE,
        )
    gan.generator.load_state_dict(torch.load(os.path.join("results", "checkpoints", "gen.00499.pt"), map_location=torch.device('cpu')))
    save_images(gan, test_noise,
            os.path.join("results", "generated", f"gen.{i:04d}.png"))
	
	#images = gan.generate_samples()
    #ims = tv.utils.make_grid(images, normalize=True)
    #plt.imshow(ims.numpy().transpose((1,2,0)))
    #plt.show()
	

if __name__ == "__main__":
    main()


	#chpt_path = os.path.join("results", "checkpoints", "gen.00499.pt")
	#gan.load_state_dict(torch.load(chpt_path))
    #images = gan.generate_samples()
    #ims = tv.utils.make_grid(images, normalize=True)
    #plt.imshow(ims.numpy().transpose((1,2,0)))
    #plt.show()
