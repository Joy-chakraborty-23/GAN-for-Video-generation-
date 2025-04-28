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
