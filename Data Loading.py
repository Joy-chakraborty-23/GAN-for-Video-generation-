transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = VideoDataset(video_folder="/content/drive/MyDrive/New folder (2)", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
