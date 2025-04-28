import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
frame_size = (256, 256)  # Resize frames to 256x256 for higher resolution
channels = 3  # RGB
num_epochs = 10
batch_size = 4000
learning_rate = 0.02

# Dataset for video frames
class VideoDataset(Dataset):
    def __init__(self, video_folder, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.frames = []
        self._load_frames()

    def _load_frames(self):
        for video_file in os.listdir(self.video_folder):
            video_path = os.path.join(self.video_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, frame_size)
                self.frames.append(frame)
            cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, channels * frame_size[0] * frame_size[1]),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), channels, frame_size[0], frame_size[1])
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(channels * frame_size[0] * frame_size[1], 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Load dataset
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = VideoDataset(video_folder="/content/drive/MyDrive/New folder (2)", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
for epoch in range(num_epochs):
    for i, imgs in enumerate(dataloader):
        valid = torch.ones(imgs.size(0), 1, device=device, dtype=torch.float)
        fake = torch.zeros(imgs.size(0), 1, device=device, dtype=torch.float)

        real_imgs = imgs.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Generate video
output_frames = []
num_generated_frames = 100
for _ in range(num_generated_frames):
    z = torch.randn(1, latent_dim, device=device)
    gen_img = generator(z).detach().cpu().numpy()
    gen_img = (gen_img * 0.5 + 0.5).clip(0, 1)
    gen_img = np.transpose(gen_img[0], (1, 2, 0)) * 255
    output_frames.append(gen_img.astype(np.uint8))

out = cv2.VideoWriter('generated_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, frame_size)
for frame in output_frames:
    out.write(frame)
out.release()

print("Video generated and saved as 'generated_video.avi'")
