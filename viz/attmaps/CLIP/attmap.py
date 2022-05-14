import os, argparse

import numpy as np
import torch, torchvision
import clip
from PIL import Image

import matplotlib.pyplot as plt


if __name__ == '__main__' :
    vit_archs = ['ViT-B/16', 'ViT-B/32', 'ViT-L/14']
    patch_sizes = {arch: int(arch[-2:]) for arch in vit_archs}

    parser = argparse.ArgumentParser('Visualize self-attention maps')
    parser.add_argument('--arch', default='ViT-B/16', type=str,
        choices=vit_archs, help='ViT-arch.')
    
    parser.add_argument("--image_path", type=str, help="Path to image")
    parser.add_argument("--output_dir", default=".", type=str, help="Path to output directory")
    
    args = parser.parse_args()
    args.patch_size = patch_sizes[args.arch]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, image_preprocessor = clip.load(args.arch, device=device)

    assert args.image_path is not None, "Enter input image"

    image = image_preprocessor(Image.open(args.image_path)).unsqueeze(0).to(device)

    # captions = ["a dog", "a cat", "a puppy", "an elephant", "a panda", "a squirrel"]
    # text = clip.tokenize(captions).to(device)

    with torch.no_grad():
        _, attn_maps = clip_model.visual.get_attention_weights(image)

    for layer, img_attn_wts in enumerate(attn_maps) :
        num_heads    = img_attn_wts.shape[1]
        img_attn_wts = img_attn_wts[0, :, 0, 1:]

        attn_map = img_attn_wts.reshape(num_heads, 14, 14)

        attn_map = torch.nn.functional.interpolate(attn_map.unsqueeze(0), scale_factor=args.patch_size,
            mode="nearest")[0] # .clamp(min=0, max=1)
        attn_map = attn_map.cpu().numpy()
        torchvision.utils.save_image(torchvision.utils.make_grid(image, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))

        os.makedirs(args.output_dir, exist_ok=True)

        fname = os.path.join(args.output_dir, f"layer-{layer}.png")
        plt.imsave(fname=fname, arr=np.concatenate(attn_map, 1), format='png')
        print(f"{fname} saved.")
