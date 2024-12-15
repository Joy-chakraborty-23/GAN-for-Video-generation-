# GAN-for-Video-generation-
This Python script implements a Generative Adversarial Network (GAN) for generating video frames. It uses PyTorch for deep learning and OpenCV for video processing. The VideoDataset class loads video frames from a folder and preprocesses them. The GAN consists of a Generator that creates high-resolution synthetic frames from random noise and a Discriminator that distinguishes real frames from generated ones. The models are trained using the Binary Cross-Entropy Loss for 10 epochs with a batch size of 4000. After training, the generator produces 100 frames, which are compiled into a video saved as generated_video.avi.
