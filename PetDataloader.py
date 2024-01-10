import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import v2
import torch

test_data = torchvision.datasets.OxfordIIITPet(root='/OxfordIIIPet', split="test", target_types="segmentation", download=True,
                                               transforms= v2.Compose([
                                                            v2.Resize(256),
                                                            v2.CenterCrop(224),
                                                            v2.ToTensor()]))

# let's create a DataLoader to easily iterate over this dataset

bs = 4
dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    imgs = sample_batched[0]
    segs = sample_batched[1]

    rows, cols = bs, 2
    figure = plt.figure(figsize=(10, 10))

    for i in range(0, bs):
        figure.add_subplot(rows, cols, 2*i+1)
        plt.title('image')
        plt.axis("off")
        plt.imshow(imgs[i].numpy().transpose(1, 2, 0))

        figure.add_subplot(rows, cols, 2*i+2)
        plt.title('seg')
        plt.axis("off")
        plt.imshow(segs[i].numpy().transpose(1, 2, 0), cmap="gray")
    plt.show()
    # display the first 3 batches
    if i_batch == 2:
        break
