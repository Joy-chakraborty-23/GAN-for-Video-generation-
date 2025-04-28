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
