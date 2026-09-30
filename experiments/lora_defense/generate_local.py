"""Generate the anti-LoRA test's images locally from the LoRAs trained on Kaggle.

Images of the people are made here, not on Colab or Kaggle. The settings are the first
run's (``build_notebook.py``): four prompts times four seeds, 25 DPM-Solver steps,
guidance 7.5, plus a ``base`` control with no LoRA. Each LoRA (r = alpha = 8 on the
attention projections) is merged into a fresh copy of the UNet's weights, so peft is
not needed. Images already on disk are skipped, so a rerun resumes.

    project/.venv/bin/python experiments/lora_defense/generate_local.py \
        --loras experiments/lora_defense/kaggle_out_v2/lora \
        --output experiments/lora_defense/kaggle_out_v2/generated
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
PROMPTS = ['a photo of sks person', 'a close-up portrait photo of sks person',
           'a photo of sks person smiling', 'a photo of sks person wearing a suit']
SEEDS = [0, 1, 2, 3]
RANK, ALPHA = 8, 8


def merged(original: dict[str, torch.Tensor], lora: dict[str, torch.Tensor]) -> dict:
    """Return the UNet weights with the LoRA folded in: W + alpha / r * B @ A."""
    weights = dict(original)
    for key in lora:
        if not key.endswith('.lora_A.weight'):
            continue
        module = key[:-len('.lora_A.weight')]
        target = f'{module}.weight'
        down = lora[key].to(original[target].device, torch.float32)
        up = lora[f'{module}.lora_B.weight'].to(original[target].device, torch.float32)
        weights[target] = (original[target].float() + ALPHA / RANK * up @ down).to(
            original[target].dtype)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loras', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    # The VAE is the float32 one, as in the training script.
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder='vae')
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, vae=vae, variant='fp16', torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.to(torch.float32)
    pipe.to('mps')
    pipe.set_progress_bar_config(disable=True)
    original = {k: v.clone() for k, v in pipe.unet.state_dict().items()}

    jobs = [('base', 'none', None)] + [
        (path.parent.name, path.stem, path)
        for path in sorted(args.loras.glob('*/*.pt')) if '.partial' not in path.name]
    for variant, identity, path in jobs:
        out_dir = args.output / variant / identity
        todo = [(p, prompt, seed) for p, prompt in enumerate(PROMPTS) for seed in SEEDS
                if not (out_dir / f'{p}_{seed}.png').exists()]
        if not todo:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        lora = torch.load(path, map_location='cpu')['weights'] if path else {}
        pipe.unet.load_state_dict(merged(original, lora))
        started = time.time()
        for p, prompt, seed in todo:
            latents = pipe(prompt, num_inference_steps=25, guidance_scale=7.5,
                           generator=torch.Generator('cpu').manual_seed(seed),
                           output_type='latent').images
            with torch.no_grad():
                image = pipe.vae.decode(latents.float() / pipe.vae.config.scaling_factor).sample
            pipe.image_processor.postprocess(image, output_type='pil')[0].save(
                out_dir / f'{p}_{seed}.png')
        print(f'{variant}/{identity}: {len(todo)} images in {time.time() - started:.0f}s',
              flush=True)
    print('generation done')


if __name__ == '__main__':
    main()
