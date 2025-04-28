import torch
from models import Generator
import cv2
import numpy as np

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
frame_size = (256, 256)
channels = 3

# Load trained generator
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

def generate_video(num_frames=100, output_file='generated_video.avi'):
    output_frames = []
    
    with torch.no_grad():
        for _ in range(num_frames):
            # Generate random noise
            z = torch.randn(1, latent_dim, device=device)
            
            # Generate frame
            gen_img = generator(z).cpu().numpy()
            
            # Convert from [-1,1] to [0,255]
            gen_img = (gen_img * 0.5 + 0.5).clip(0, 1)
            gen_img = np.transpose(gen_img[0], (1, 2, 0)) * 255
            output_frames.append(gen_img.astype(np.uint8))
    
    # Save as video
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 30, frame_size)
    for frame in output_frames:
        out.write(frame)
    out.release()
    
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    generate_video()
