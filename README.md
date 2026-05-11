# SigMa


**SigMa enables pre-trained diffusion transformers to generate ultra-high-resolution images far beyond their training scale.** It dynamically adjusts positional encodings according to both image scale and denoising timestep, matching the model's evolving frequency needs during sampling and achieving faithful 4K × 4K generation without retraining or extra sampling cost.

<img src="docs/assets.png" alt="SigMa Results" width="100%">

## Installation

```bash
conda create -n sigma python=3.10
conda activate sigma
pip install -r requirements.txt
```

## Usage

```bash
python run_sigma.py --prompt "Your text prompt here"
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | Dark fantasy scene | Text prompt for image generation |
| `--height` | `1024` | Image height in pixels |
| `--width` | `1024` | Image width in pixels |
| `--steps` | `28` | Number of inference steps |
| `--seed` | `42` | Random seed |
| `--method` | `yarn` | Position encoding backend: `yarn`, `ntk`, or `base` |
| `--no_sigma` | `False` | Disable SigMa and run the underlying positional method directly |
| `--output_dir` | `outputs` | Directory for generated images |

Example:

```bash
# 4K SigMa example
python run_sigma.py \
  --height 4096 \
  --width 4096 \
  --prompt "A silhouetted pagoda stands against a large red moon, surrounded by dark mountains and trees, with some stylized birds flying in the sky. Stars twinkle in the night backdrop."
```

Generated images are saved to the selected `output_dir`.

