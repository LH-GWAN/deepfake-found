"""Write ``lora_defense.ipynb``: the GPU half of the LoRA-defence experiment.

The notebook is generated rather than hand-edited so its code can be read,
linted and diffed as ordinary Python. Run it on Colab with a GPU runtime.

    python experiments/lora_defense/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """\
# DeepShield: does perturbing a photograph stop LoRA fine-tuning on it?

For three people, a Stable Diffusion 1.5 LoRA is trained on eight photographs, once per
variant of those photographs, and then made to generate the person:

| variant | what was done to the photographs |
|---|---|
| `clean` | nothing |
| `arcface8` | the face-swap defence (8/255 against ArcFace), made locally |
| `encoder8`, `encoder16` | PhotoGuard encoder attack: VAE latent pulled towards grey |
| `fsmg8`, `fsmg16` | Anti-DreamBooth FSMG: training loss pushed up on the fixed model |

A `base` control generates the same prompts with no LoRA at all.

**Checkpoints.** Everything is written to `MyDrive/deepshield_lora_defense/` as it is produced,
and every stage skips work that is already there. If the runtime disconnects or the GPU
quota runs out, reconnect later and run all cells again; it resumes where it stopped.

**On the first run** the setup cell asks for `faces.zip` (made locally by
`scripts/prepare_lora_defense.py`) and keeps it in `MyDrive/deepshield_lora_defense/`.

The photographs are LFW images of public figures, used here only for this measurement.
Nothing generated here is to be published.
"""

SETUP = """\
import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
RUN = '/content/drive/MyDrive/deepshield_lora_defense'
os.makedirs(RUN, exist_ok=True)
if not os.path.exists(f'{RUN}/faces.zip'):
    import shutil
    from google.colab import files
    uploaded = files.upload()                      # choose faces.zip
    name = next(iter(uploaded))
    shutil.move(name, f'{RUN}/faces.zip')
print('faces.zip on Drive:', os.path.getsize(f'{RUN}/faces.zip') // 1024, 'KB')
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                     capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers>=0.30', 'peft>=0.11'],
               check=True)
# Colab ships a torchao older than peft accepts, and peft refuses to add an adapter
# while it is installed. Nothing here uses it.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)
"""

MODELS = """\
import glob, io, json, math, random, time, zipfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import (AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

DEV = 'cuda'
MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
RES = 512
PROMPT = 'a photo of sks person'
IDENTITIES = ['tom_hanks', 'jennifer_lopez', 'hugo_chavez']
PROTECTED = f'{RUN}/protected_s60'   # versioned: perturbations made with 60 fp16 PGD steps

tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(
    MODEL, subfolder='text_encoder', variant='fp16', torch_dtype=torch.float16).to(DEV)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder='vae', variant='fp16').to(DEV).float()
noise_scheduler = DDPMScheduler.from_pretrained(MODEL, subfolder='scheduler')
for module in (text_encoder, vae):
    module.requires_grad_(False)

def load_unet():
    unet = UNet2DConditionModel.from_pretrained(
        MODEL, subfolder='unet', variant='fp16', torch_dtype=torch.float16).to(DEV)
    unet.requires_grad_(False)
    return unet

@torch.no_grad()
def encode_prompt(text):
    ids = tokenizer([text], padding='max_length', max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors='pt').input_ids.to(DEV)
    return text_encoder(ids)[0]

COND = encode_prompt(PROMPT)

with zipfile.ZipFile(f'{RUN}/faces.zip') as bundle:
    bundle.extractall('/content/faces')

def photos(variant, identity):
    root = PROTECTED if variant not in ('clean', 'arcface8') else '/content/faces'
    paths = sorted(glob.glob(f'{root}/{variant}/{identity}/*.png'),
                   key=lambda p: int(os.path.basename(p)[:-4]))
    return paths

def to_tensor(path):
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEV)

def upscale(x):
    return F.interpolate(x, size=(RES, RES), mode='bicubic', align_corners=False).clamp(0, 1)

print('models ready')
"""

PROTECT = """\
# Perturbations are made at the posted resolution (250 px) and reach the model through
# the same upscaling a training script applies, so what is saved is what was optimised.
# The models run in fp16 under autocast: a T4 is several times slower in fp32, and a
# sign-of-gradient step does not need more precision than that.
STEPS = 60
GREY_LATENT = None

def encoder_loss(x, unet):
    global GREY_LATENT
    if GREY_LATENT is None:
        with torch.no_grad():
            GREY_LATENT = vae.encode(torch.zeros(1, 3, RES, RES, device=DEV)).latent_dist.mean
    with torch.autocast('cuda', dtype=torch.float16):
        latent = vae.encode(upscale(x) * 2 - 1).latent_dist.mean
    return -F.mse_loss(latent.float(), GREY_LATENT)          # ascend: move towards grey

def fsmg_loss(x, unet):
    with torch.autocast('cuda', dtype=torch.float16):
        latent = vae.encode(upscale(x) * 2 - 1).latent_dist.sample()
        latent = latent.float() * vae.config.scaling_factor
        noise = torch.randn_like(latent)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=DEV)
        noisy = noise_scheduler.add_noise(latent, noise, t)
        pred = unet(noisy.half(), t, COND).sample.float()
    return F.mse_loss(pred, noise)                   # ascend: make the model fail

def perturb(path, loss_fn, epsilon, unet):
    x = to_tensor(path)
    delta = torch.zeros_like(x, requires_grad=True)
    step = epsilon / 10
    for _ in range(STEPS):
        loss = loss_fn((x + delta).clamp(0, 1), unet)
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta += step * grad.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_((x + delta).clamp(0, 1) - x)
    out = ((x + delta).clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
    return Image.fromarray(np.round(out).astype(np.uint8))

ATTACKS = {'encoder8': (encoder_loss, 8), 'encoder16': (encoder_loss, 16),
           'fsmg8': (fsmg_loss, 8), 'fsmg16': (fsmg_loss, 16)}
unet = load_unet()
for variant, (loss_fn, eps) in ATTACKS.items():
    for identity in IDENTITIES:
        out_dir = f'{PROTECTED}/{variant}/{identity}'
        os.makedirs(out_dir, exist_ok=True)
        for path in photos('clean', identity):
            target = f'{out_dir}/{os.path.basename(path)}'
            if os.path.exists(target):
                continue
            started = time.time()
            perturb(path, loss_fn, eps / 255, unet).save(target)
            print(f'{variant} {identity} {os.path.basename(path)} {time.time() - started:.1f}s',
                  flush=True)
del unet
torch.cuda.empty_cache()
print('perturbations done')
"""

TRAIN = """\
TRAIN_STEPS = 400
CHECKPOINT_EVERY = 100
RANK = 8

def latents_of(paths):
    # both horizontal flips, as the usual DreamBooth augmentation
    means, stds = [], []
    with torch.no_grad():
        for path in paths:
            x = upscale(to_tensor(path)) * 2 - 1
            for view in (x, torch.flip(x, dims=[3])):
                dist = vae.encode(view).latent_dist
                means.append(dist.mean)
                stds.append(dist.std)
    return torch.cat(means), torch.cat(stds)

def train_lora(variant, identity):
    final = f'{RUN}/lora/{variant}/{identity}.pt'
    partial = f'{RUN}/lora/{variant}/{identity}.partial.pt'
    if os.path.exists(final):
        return
    os.makedirs(os.path.dirname(final), exist_ok=True)
    means, stds = latents_of(photos(variant, identity))
    unet = load_unet()
    unet.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights='gaussian',
                                target_modules=['to_k', 'to_q', 'to_v', 'to_out.0']))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    start = 0
    if os.path.exists(partial):
        saved = torch.load(partial)
        set_peft_model_state_dict(unet, saved['weights'])
        start = saved['step']
        print(f'resuming {variant}/{identity} at step {start}')
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    generator = torch.Generator(device=DEV).manual_seed(start)
    unet.train()
    for step in range(start, TRAIN_STEPS):
        i = int(torch.randint(0, means.shape[0], (1,), generator=generator, device=DEV))
        latent = (means[i:i+1] + stds[i:i+1] * torch.randn(
            means[i:i+1].shape, generator=generator, device=DEV)) * vae.config.scaling_factor
        noise = torch.randn(latent.shape, generator=generator, device=DEV)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,),
                          generator=generator, device=DEV)
        noisy = noise_scheduler.add_noise(latent, noise, t)
        with torch.autocast('cuda', dtype=torch.float16):
            pred = unet(noisy.half(), t, COND).sample
        loss = F.mse_loss(pred.float(), noise)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % CHECKPOINT_EVERY == 0 and step + 1 < TRAIN_STEPS:
            torch.save({'weights': get_peft_model_state_dict(unet), 'step': step + 1}, partial)
    torch.save({'weights': get_peft_model_state_dict(unet), 'step': TRAIN_STEPS}, final)
    if os.path.exists(partial):
        os.remove(partial)
    del unet, optimizer
    torch.cuda.empty_cache()

VARIANTS = ['clean', 'arcface8', 'encoder8', 'encoder16', 'fsmg8', 'fsmg16']
for variant in VARIANTS:
    for identity in IDENTITIES:
        started = time.time()
        train_lora(variant, identity)
        print(f'lora {variant}/{identity} ready ({time.time() - started:.0f}s)', flush=True)
print('training done')
"""

GENERATE = """\
PROMPTS = ['a photo of sks person', 'a close-up portrait photo of sks person',
           'a photo of sks person smiling', 'a photo of sks person wearing a suit']
SEEDS = [0, 1, 2, 3]

def pipeline_with(unet):
    pipe = StableDiffusionPipeline(
        vae=vae.half(), text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder='scheduler'),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def generate(variant, identity, unet):
    out_dir = f'{RUN}/generated/{variant}/{identity}'
    os.makedirs(out_dir, exist_ok=True)
    pipe = None
    for p, prompt in enumerate(PROMPTS):
        for seed in SEEDS:
            target = f'{out_dir}/{p}_{seed}.png'
            if os.path.exists(target):
                continue
            if pipe is None:
                pipe = pipeline_with(unet)
            image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5,
                         generator=torch.Generator(device=DEV).manual_seed(seed)).images[0]
            image.save(target)

unet = load_unet()
generate('base', 'none', unet)
del unet
for variant in VARIANTS:
    for identity in IDENTITIES:
        unet = load_unet()
        unet.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights='gaussian',
                                    target_modules=['to_k', 'to_q', 'to_v', 'to_out.0']))
        saved = torch.load(f'{RUN}/lora/{variant}/{identity}.pt')
        set_peft_model_state_dict(unet, saved['weights'])
        unet.eval()
        generate(variant, identity, unet)
        del unet
        torch.cuda.empty_cache()
        print(f'generated {variant}/{identity}', flush=True)
vae.float()
print('generation done')
"""

PACK = """\
# One file to bring back: generated images plus the perturbed photographs.
with zipfile.ZipFile(f'{RUN}/results.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in glob.glob(f'{RUN}/generated/**/*.png', recursive=True):
        bundle.write(path, os.path.relpath(path, RUN))
    for path in glob.glob(f'{PROTECTED}/**/*.png', recursive=True):
        bundle.write(path, 'protected/' + os.path.relpath(path, PROTECTED))
print('wrote', f'{RUN}/results.zip', os.path.getsize(f'{RUN}/results.zip') // 1024, 'KB')
"""


def cell(kind: str, source: str) -> dict:
    """Return one notebook cell."""
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": lines,
    }


def main() -> None:
    """Write the notebook next to this file."""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell("markdown", INTRO),
            cell("code", SETUP),
            cell("code", MODELS),
            cell("code", PROTECT),
            cell("code", TRAIN),
            cell("code", GENERATE),
            cell("code", PACK),
        ],
    }
    target = HERE / "lora_defense.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
