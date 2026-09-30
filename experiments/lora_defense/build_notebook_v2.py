"""Write ``lora_defense_v2.ipynb``: published anti-personalisation tools against LoRA.

The first run (``build_notebook.py``) tried PhotoGuard's encoder attack and
Anti-DreamBooth's weaker FSMG at 60 PGD steps; nothing stopped a LoRA from
learning the person. This run tries the published tools as their authors ship
them, on the same photographs and the same LoRA recipe:

``mist``, ``mist16``
    Mist v2 (psyker-team/mist-v2), which alternates PGD with LoRA fine-tuning
    and was verified by its authors against LoRA; at its default budget
    (8/255 of the [-1, 1] range) and at 16/255 of that range, the 8/255 of
    [0, 1] the other variants used.
``aspl``
    Anti-DreamBooth ASPL (VinAIResearch/Anti-DreamBooth), the stronger of its
    two methods, at the repository's script settings.

Both tools pin old libraries, so each runs in its own Python 3.10 virtual
environment. The surrogate model is Stable Diffusion 1.5, the model the LoRA
is trained on: the defence's best case.

Only perturbation and LoRA training run here. Images of the people are
generated from the trained LoRAs locally, not on Colab.

    python experiments/lora_defense/build_notebook_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """\
# DeepShield: do published anti-LoRA tools stop LoRA fine-tuning?

Variants: `clean`, `mist` (Mist v2 default), `mist16` (Mist v2 at 16/255 of [-1, 1]),
`aspl` (Anti-DreamBooth ASPL). Surrogate and LoRA base: Stable Diffusion 1.5.

**Checkpoints.** Everything goes to `MyDrive/deepshield_lora_defense_v2/` as it is made,
and every stage skips finished work, so after a disconnect or a quota stop, run all cells
again later on the same account.

Photographs are LFW images of public figures, used only for this measurement. This
notebook makes perturbed photographs and trains LoRAs; it generates no images of the people.
"""

SETUP = """\
import os, subprocess, sys, glob, shutil, zipfile, time
from google.colab import drive
drive.mount('/content/drive')
RUN = '/content/drive/MyDrive/deepshield_lora_defense_v2'
os.makedirs(RUN, exist_ok=True)
if not os.path.exists(f'{RUN}/faces.zip'):
    shutil.copy('/content/faces.zip', f'{RUN}/faces.zip')   # uploaded through the Files pane
print('faces.zip on Drive:', os.path.getsize(f'{RUN}/faces.zip') // 1024, 'KB')
with zipfile.ZipFile(f'{RUN}/faces.zip') as bundle:
    bundle.extractall('/content/faces')
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                     capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers>=0.30', 'peft>=0.11',
                'uv'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)
IDENTITIES = ['tom_hanks', 'jennifer_lopez', 'hugo_chavez']
MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
LOG = f'{RUN}/log.txt'

def log(message):
    stamp = time.strftime('%H:%M:%S')
    print(f'{stamp} {message}', flush=True)
    with open(LOG, 'a') as handle:
        handle.write(f'{stamp} {message}\\n')

def sh(command, logfile=None):
    '''Run a shell command, tee its output to a Drive log, raise on failure.'''
    target = logfile or f'{RUN}/shell.log'
    with open(target, 'a') as handle:
        handle.write(f'$ {command}\\n')
        done = subprocess.run(command, shell=True, stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode:
        tail = open(target).read()[-3000:]
        raise RuntimeError(f'failed ({done.returncode}): {command}\\n{tail}')

def local_dir(identity):
    '''The clean photographs of one identity as a folder of images only.'''
    out = f'/content/input/{identity}'
    if not os.path.isdir(out):
        os.makedirs(out)
        for path in glob.glob(f'/content/faces/clean/{identity}/*.png'):
            shutil.copy(path, out)
    return out

log('setup done')
"""

ENVS = """\
# One Python 3.10 environment per tool, with the libraries each repository pins.
# They live on the VM disk and are rebuilt after a disconnect (a few minutes each).
# Newer setuptools dropped pkg_resources, which old setup.py files (and accelerate
# 0.16 at run time) still import; keep builds and both venvs on an older one.
os.chdir('/content')
open('/content/build_constraints.txt', 'w').write('setuptools<70\\n')
os.environ['UV_BUILD_CONSTRAINT'] = '/content/build_constraints.txt'
if not os.path.isdir('/content/mist-v2'):
    sh('git clone -q https://github.com/psyker-team/mist-v2.git /content/mist-v2')
if not os.path.isdir('/content/Anti-DreamBooth'):
    sh('git clone -q https://github.com/VinAIResearch/Anti-DreamBooth.git /content/Anti-DreamBooth')
if not os.path.exists('/content/venv_mist/bin/python'):
    sh('uv venv -q --python 3.10 /content/venv_mist')
    # torch 2.0.1 needs triton, which needs lit; the cu118 index only has lit as an
    # sdist that fails to build under uv, so the PyPI wheel goes in first.
    sh('uv pip install -q --python /content/venv_mist/bin/python lit')
    sh('uv pip install -q --python /content/venv_mist/bin/python '
       'torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118')
    sh('uv pip install -q --python /content/venv_mist/bin/python '
       '"diffusers==0.21.4" "transformers==4.33.3" "accelerate==0.23.0" "huggingface_hub==0.17.3" '
       'ftfy tqdm scipy fire safetensors opencv-python-headless colorama torchmetrics '
       'xformers==0.0.22 "numpy<2" pillow "datasets==2.14.6" "pyarrow<15" pynvml tensorboard Jinja2 '
       'git+https://github.com/cloneofsimo/lora.git "setuptools<70"')
    log('mist environment ready')
if not os.path.exists('/content/venv_aspl/bin/python'):
    sh('uv venv -q --python 3.10 /content/venv_aspl')
    # The PyPI build of torch 1.13.1 is cu117 and brings CUDA 11 runtime libraries as
    # packages; bitsandbytes 0.41.1 ships no cu116 binary, so the repo's cu116 is swapped.
    sh('uv pip install -q --python /content/venv_aspl/bin/python '
       'torch==1.13.1 torchvision==0.14.1 xformers==0.0.16 nvidia-cusparse-cu11')
    sh('uv pip install -q --python /content/venv_aspl/bin/python '
       '"diffusers==0.13.1" "transformers==4.26.0" "accelerate==0.16.0" "huggingface_hub==0.13.4" '
       '"datasets==2.10.1" ftfy tqdm tensorboard Jinja2 "numpy<2" bitsandbytes==0.41.1 scipy '
       'safetensors pillow "pyarrow<15" "setuptools<70"')
    log('aspl environment ready')
# bitsandbytes looks up the CUDA 11 libraries on LD_LIBRARY_PATH, not in the venv.
NVIDIA = '/content/venv_aspl/lib/python3.10/site-packages/nvidia'
if not os.path.exists(f'{NVIDIA}/cuda_runtime/lib/libcudart.so'):
    os.symlink(f'{NVIDIA}/cuda_runtime/lib/libcudart.so.11.0', f'{NVIDIA}/cuda_runtime/lib/libcudart.so')
ASPL_LIBS = ':'.join(f'{NVIDIA}/{name}/lib' for name in ('cuda_runtime', 'cublas', 'cusparse'))
"""

CLASS = """\
# Prior-preservation images of a generic person, shared by both tools. Nobody in particular.
CLASS_DIR = f'{RUN}/class_person'
os.makedirs(CLASS_DIR, exist_ok=True)
NEEDED = 200
have = len(glob.glob(f'{CLASS_DIR}/*.png'))
if have < NEEDED:
    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                   safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    for index in range(have, NEEDED):
        image = pipe('a photo of person', num_inference_steps=25,
                     generator=torch.Generator('cuda').manual_seed(index)).images[0]
        image.save(f'{CLASS_DIR}/{index}.png')
    del pipe
    torch.cuda.empty_cache()
log(f'class images: {len(glob.glob(CLASS_DIR + "/*.png"))}')
"""

PROTECT = """\
PROTECTED = f'{RUN}/protected'

# The tools' old huggingface_hub versions can no longer fetch from the Hub, so they get
# a local copy of the model downloaded with this kernel's current one.
SD_DIR = '/content/sd15'
if not os.path.exists(f'{SD_DIR}/model_index.json'):
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, local_dir=SD_DIR, allow_patterns=[
        'model_index.json', '*/*.json', '*/*.txt', 'unet/diffusion_pytorch_model.safetensors',
        'vae/diffusion_pytorch_model.safetensors', 'text_encoder/model.safetensors',
        'safety_checker/model.safetensors'])

def finished(variant, identity):
    return len(glob.glob(f'{PROTECTED}/{variant}/{identity}/*.png')) >= 8

def collect(found, variant, identity):
    '''Copy a tool's output images to <variant>/<identity>/<n>.png, n from the source name.'''
    out = f'{PROTECTED}/{variant}/{identity}'
    os.makedirs(out, exist_ok=True)
    for path in found:
        number = os.path.basename(path).split('_')[-1].split('.')[0]
        shutil.copy(path, f'{out}/{number}.png')

MIST = {'mist': 8, 'mist16': 16}
for variant, eps in MIST.items():
    for identity in IDENTITIES:
        if finished(variant, identity):
            continue
        work = f'/content/work/{variant}/{identity}'
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        started = time.time()
        sh(f'cd /content/mist-v2 && /content/venv_mist/bin/accelerate launch attacks/mist.py '
           f'--cuda --low_vram_mode --pretrained_model_name_or_path {SD_DIR} '
           f'--instance_data_dir {local_dir(identity)} --output_dir {work} '
           f'--class_data_dir {CLASS_DIR} --instance_prompt "a photo of sks person" '
           f'--class_prompt "a photo of person" --mixed_precision fp16 '
           f'--pgd_eps {eps / 255}', f'{RUN}/mist.log')
        collect(sorted(glob.glob(f'{work}/**/*.png', recursive=True)), variant, identity)
        log(f'{variant} {identity} done in {time.time() - started:.0f}s')

# aspl.py as published does not fit a T4 (12 GB RAM, 15 GB GPU); these changes
# leave its results alone:
# - it deep-copies the fp32 UNet and text encoder on the CPU twice in its first round,
#   overflowing RAM; train_one_epoch moves them to the GPU in bf16 anyway, so they go
#   there before the loop;
# - train_one_epoch deep-copies the models it is given, though its callers pass a fresh
#   copy or discard the original, so it trains them in place;
# - gradient checkpointing recomputes activations instead of storing them;
# - zero_grad frees the gradients instead of keeping zero-filled copies of both models;
# - the PGD step runs its eight images through the VAE and UNet two at a time, the loss
#   being a mean over images so the slices add up to the same gradient, and freezes the
#   surrogate's weights, which are thrown away after the step, so no weight gradients
#   are kept.
# Colab sessions end before an identity's 50 rounds (about 2.3 hours) are done, so
# every checkpoint also saves the surrogate weights and the perturbed images to Drive
# and a restarted run picks up from there.
# xformers has no bf16 backward for SD 1.5's 40-wide heads on a T4, so it stays off.
ASPL_PY = '/content/Anti-DreamBooth/attacks/aspl.py'
sh('cd /content/Anti-DreamBooth && git checkout -q attacks/aspl.py')
code = open(ASPL_PY).read()
for old, new in [
    ('    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])\\n',
     '    unet, text_encoder = models[0], models[1]\\n'),
    ('    f = [unet, text_encoder]\\n',
     '    unet.enable_gradient_checkpointing()\\n'
     '    text_encoder.gradient_checkpointing_enable()\\n'
     "    f = [unet.to('cuda', dtype=torch.bfloat16), "
     "text_encoder.to('cuda', dtype=torch.bfloat16)]\\n"),
]:
    assert code.count(old) == 1, old
    code = code.replace(old, new)
code = code.replace('.zero_grad()', '.zero_grad(set_to_none=True)')
PGD_SLICES = '''        assert target_tensor is None
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        grad = torch.zeros_like(perturbed_images)
        loss_value = 0.0
        for lo in range(0, len(perturbed_images), 2):
            part = perturbed_images[lo:lo + 2].detach().clone().requires_grad_(True)
            latents = vae.encode(part.to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids[lo:lo + 2].to(device))[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            part_loss = (F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                         * len(part) / len(perturbed_images))
            part_loss.backward()
            grad[lo:lo + 2] = part.grad
            loss_value += part_loss.item()
        loss = torch.tensor(loss_value)
'''
start = code.index('        perturbed_images.requires_grad = True\\n')
end = code.index('        loss.backward()\\n', start) + len('        loss.backward()\\n')
assert code.count('perturbed_images.grad.sign()') == 1
code = code[:start] + PGD_SLICES + code[end:]
code = code.replace('perturbed_images.grad.sign()', 'grad.sign()')
RESUME = '''    start = 0
    resume = os.environ.get('ASPL_RESUME')
    if resume and os.path.exists(resume):
        saved = torch.load(resume, map_location='cuda')
        f[0].load_state_dict(saved['unet'])
        f[1].load_state_dict(saved['text_encoder'])
        perturbed_data = saved['perturbed'].cpu()
        start = saved['done']
        del saved
        print(f"Resumed after round {start} from {resume}", flush=True)
    for i in range(start, args.max_train_steps):
'''
SAVE = '''            if resume:
                torch.save({'done': i + 1, 'unet': f[0].state_dict(),
                            'text_encoder': f[1].state_dict(),
                            'perturbed': perturbed_data.detach().cpu()}, resume + '.part')
                os.replace(resume + '.part', resume)
'''
for old, new in [('    for i in range(args.max_train_steps):\\n', RESUME),
                 ('            print(f"Saved noise at step {i+1} to {save_folder}")\\n',
                  '            print(f"Saved noise at step {i+1} to {save_folder}")\\n' + SAVE)]:
    assert code.count(old) == 1, old
    code = code.replace(old, new)
open(ASPL_PY, 'w').write(code)

def aspl(identity, eight_bit):
    work = f'/content/work/aspl/{identity}'
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    if eight_bit:
        sh("cd /content/Anti-DreamBooth && sed -i 's/torch.optim.AdamW(/"
           "__import__(\\"bitsandbytes\\").optim.AdamW8bit(/' attacks/aspl.py")
    os.makedirs(f'{RUN}/aspl_resume', exist_ok=True)
    sh(f'cd /content/Anti-DreamBooth && LD_LIBRARY_PATH={ASPL_LIBS}:$LD_LIBRARY_PATH '
       f'ASPL_RESUME={RUN}/aspl_resume/{identity}.pt '
       f'/content/venv_aspl/bin/accelerate launch '
       f'--mixed_precision fp16 attacks/aspl.py '
       f'--pretrained_model_name_or_path={SD_DIR} '
       f'--instance_data_dir_for_train={local_dir(identity)} '
       f'--instance_data_dir_for_adversarial={local_dir(identity)} '
       f'--instance_prompt="a photo of sks person" --class_data_dir={CLASS_DIR} '
       f'--num_class_images=200 --class_prompt="a photo of person" --output_dir={work} '
       f'--center_crop --with_prior_preservation --prior_loss_weight=1.0 --resolution=512 '
       f'--train_text_encoder --train_batch_size=1 --max_train_steps=50 '
       f'--max_f_train_steps=3 --max_adv_train_steps=6 --checkpointing_iterations=5 '
       f'--learning_rate=5e-7 --pgd_alpha=5e-3 --pgd_eps=5e-2 --mixed_precision=fp16',
       f'{RUN}/aspl.log')
    return sorted(glob.glob(f'{work}/noise-ckpt/50/*.png'))

EIGHT_BIT = os.path.exists(f'{RUN}/aspl_needs_8bit')
for identity in IDENTITIES:
    if finished('aspl', identity):
        continue
    started = time.time()
    try:
        found = aspl(identity, EIGHT_BIT)
    except RuntimeError as error:
        if 'out of memory' not in str(error).lower() or EIGHT_BIT:
            raise
        open(f'{RUN}/aspl_needs_8bit', 'w').write('T4 out of memory with AdamW')
        log('aspl ran out of memory; retrying with 8-bit AdamW')
        EIGHT_BIT = True
        found = aspl(identity, True)
    collect(found, 'aspl', identity)
    if os.path.exists(f'{RUN}/aspl_resume/{identity}.pt'):
        os.remove(f'{RUN}/aspl_resume/{identity}.pt')
    log(f'aspl {identity} done in {time.time() - started:.0f}s (8-bit Adam: {EIGHT_BIT})')
log('perturbations done')
"""

TRAIN = """\
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

DEV = 'cuda'
RES = 512
PROMPT = 'a photo of sks person'
TRAIN_STEPS = 400
CHECKPOINT_EVERY = 100
RANK = 8
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(
    MODEL, subfolder='text_encoder', variant='fp16', torch_dtype=torch.float16).to(DEV)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder='vae', variant='fp16').to(DEV).float()
noise_scheduler = DDPMScheduler.from_pretrained(MODEL, subfolder='scheduler')
for module in (text_encoder, vae):
    module.requires_grad_(False)
with torch.no_grad():
    ids = tokenizer([PROMPT], padding='max_length', max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors='pt').input_ids.to(DEV)
    COND = text_encoder(ids)[0]

def photos(variant, identity):
    root = '/content/faces' if variant == 'clean' else PROTECTED
    return sorted(glob.glob(f'{root}/{variant}/{identity}/*.png'),
                  key=lambda p: int(os.path.basename(p)[:-4]))

def to_512(path):
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEV)
    return F.interpolate(x, size=(RES, RES), mode='bicubic', align_corners=False).clamp(0, 1)

def latents_of(paths):
    means, stds = [], []
    with torch.no_grad():
        for path in paths:
            x = to_512(path) * 2 - 1
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
    unet = UNet2DConditionModel.from_pretrained(
        MODEL, subfolder='unet', variant='fp16', torch_dtype=torch.float16).to(DEV)
    unet.requires_grad_(False)
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

VARIANTS = ['clean', 'mist', 'mist16', 'aspl']
for variant in VARIANTS:
    for identity in IDENTITIES:
        started = time.time()
        train_lora(variant, identity)
        log(f'lora {variant}/{identity} ready ({time.time() - started:.0f}s)')
log('training done')
"""

PACK = """\
# One small file to bring back: the LoRA weights and the perturbed photographs.
with zipfile.ZipFile(f'{RUN}/results_v2.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in glob.glob(f'{RUN}/lora/**/*.pt', recursive=True):
        if not path.endswith('.partial.pt'):
            bundle.write(path, os.path.relpath(path, RUN))
    for path in glob.glob(f'{PROTECTED}/**/*.png', recursive=True):
        bundle.write(path, os.path.relpath(path, RUN))
log(f'wrote results_v2.zip ({os.path.getsize(RUN + "/results_v2.zip") // 1024} KB)')
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
            cell("code", ENVS),
            cell("code", CLASS),
            cell("code", PROTECT),
            cell("code", TRAIN),
            cell("code", PACK),
        ],
    }
    target = HERE / "lora_defense_v2.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
