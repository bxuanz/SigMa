"""
Official implementation for ultra-high resolution image generation as presented in:
SigMa
"""

import argparse
import os

import torch

from flux.pipeline_flux import FluxPipeline
from flux.transformer_flux import FluxTransformer2DModel


def main():
    parser = argparse.ArgumentParser(description='SigMa: Generate ultra-high resolution images with FLUX')
    parser.add_argument(
        '--prompt',
        type=str,
        default="A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic buildings in the distance.",
        help='Text prompt for image generation',
    )
    parser.add_argument('--height', type=int, default=1024, help='Image height in pixels')
    parser.add_argument('--width', type=int, default=1024, help='Image width in pixels')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument(
        '--method',
        type=str,
        choices=['yarn', 'ntk', 'base'],
        default='yarn',
        help='Position encoding method (yarn, ntk, or base)',
    )
    parser.add_argument(
        '--no_sigma',
        action='store_true',
        help='Disable SigMa and use the underlying positional method directly',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save generated images',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator('cuda').manual_seed(args.seed)

    use_sigma = not args.no_sigma

    transformer = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-Krea-dev',
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        sigma=use_sigma,
        method=args.method,
    )

    pipe = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-Krea-dev',
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    print(f"Generating {args.height}x{args.width} image with {args.steps} steps...")
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=4.5,
        generator=generator,
        num_inference_steps=args.steps,
    ).images[0]

    method_name = args.method
    if use_sigma:
        method_name = f'sigma_{method_name}'

    filename = os.path.join(
        args.output_dir,
        f'seed_{args.seed}_method_{method_name}_res_{args.height}x{args.width}.png',
    )
    image.save(filename)
    print(f'✓ Image saved to: {filename}')


if __name__ == '__main__':
    main()
