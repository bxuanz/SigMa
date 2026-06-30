"""
Official implementation for ultra-high resolution image generation with Qwen
SigMa: Resolution-aware adaptive position extrapolation for ultra-high resolution diffusion
"""

import os
import argparse
import torch
from diffusers import DiffusionPipeline

from qwen.transformer_qwenimage import QwenImageTransformer2DModel


def load_prompts(prompt_file):
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description='SigMa: Generate ultra-high resolution images with Qwen'
    )

    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for image generation (single prompt mode)',
    )
    parser.add_argument(
        '--prompt_file',
        type=str,
        default=None,
        help='Path to a text file with one prompt per line (batch mode)',
    )

    parser.add_argument('--height', type=int, default=3072, help='Image height in pixels')
    parser.add_argument('--width', type=int, default=3072, help='Image width in pixels')
    parser.add_argument('--steps', type=int, default=40, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    parser.add_argument(
        '--no_sigma',
        action='store_true',
        help='Disable SigMa and use base positional encoding',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for generated images',
    )
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of prompts (0-based, inclusive)')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index of prompts (exclusive, -1 means all)')

    args = parser.parse_args()

    if args.prompt is None and args.prompt_file is None:
        args.prompt = "A woman kneels on the forest floor, smiling as she offers grapes to a large brown bear beside her, surrounded by tall birch trees."

    use_sigma = not args.no_sigma
    method_label = 'sigma-yarn' if use_sigma else 'base'

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"outputs_qwen_{method_label}_{args.height}x{args.width}"
    os.makedirs(output_dir, exist_ok=True)

    if args.prompt_file:
        all_prompts = load_prompts(args.prompt_file)
        start = args.start_idx
        end = len(all_prompts) if args.end_idx == -1 else args.end_idx
        prompts = all_prompts[start:end]
        prompt_indices = list(range(start, start + len(prompts)))
        print(f"Loaded {len(prompts)} prompts (index {start}-{start+len(prompts)-1}) from {args.prompt_file}")
    else:
        prompts = [args.prompt]
        prompt_indices = [0]

    model_name = "/nas/zbx/1/hf_weights/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    print(f"Loading {model_name} on {device}...")

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        sigma=use_sigma,
    )

    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)
    pipe.vae.enable_tiling()

    positive_magic = ", Ultra HD, 4K, cinematic composition."
    negative_prompt = " "

    for i, (prompt_idx, prompt) in enumerate(zip(prompt_indices, prompts)):
        full_prompt = prompt + positive_magic
        generator = torch.Generator(device=device).manual_seed(args.seed)

        print(f"[{i+1}/{len(prompts)}] Generating {args.height}x{args.width} with method={method_label} ...")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            true_cfg_scale=4.0,
            generator=generator,
        ).images[0]

        filename = os.path.join(output_dir, f"{prompt_idx+1:04d}.png")
        image.save(filename)
        print(f"  Saved: {filename}")

    print(f"\nDone! {len(prompts)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
