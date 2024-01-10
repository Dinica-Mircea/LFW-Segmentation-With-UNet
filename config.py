from pathlib import Path
import torch
from UNet import UNet

batch_size = 10
val_percent = 0.3
epochs = 50
learning_rate = 0.0001
save_checkpoint = True
img_scale = 0.5
amp = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=3)
#model = model.to(memory_format=torch.channels_last)
model.to(device=device)
gradient_clipping = 1.0
dir_checkpoint = Path('./checkpoints/')
