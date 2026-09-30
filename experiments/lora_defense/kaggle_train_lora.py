"""Train the anti-LoRA test's LoRAs on Kaggle, with the settings of lora_defense_v2.ipynb.

Reads the photographs from attached Kaggle datasets holding ``<variant>/<identity>/<n>.png``
folders anywhere below ``/kaggle/input`` (variant: clean, mist, mist16, aspl) and writes one
LoRA per variant and identity to ``/kaggle/working/lora/<variant>/<identity>.pt``.

Every LoRA checkpoints to ``<identity>.partial.pt`` each 100 steps and a rerun resumes
from it. If a run dies, attach its output as an input of the next version: LoRAs found
under ``/kaggle/input/*/lora`` are copied in first, so finished ones are skipped.

``STOP_AT_STEP=n`` stops the first LoRA it trains after step n, to test the resume; once a
resume has been seen (``lora_resume_tested``) it is ignored. Photos protected by
kaggle_aspl.py or kaggle_mist.py under ``/kaggle/working/protected`` are read too.

``LORA_IDENTITIES`` names the people (comma-separated) instead of the first three,
``LORA_VARIANTS`` picks the variants (comma-separated), and
``LORA_SHARD=k/n`` trains only every n-th LoRA from the k-th, so two processes can share
the two GPUs of a "GPU T4 x2" session.
"""

import glob
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
IDENTITIES = ['tom_hanks', 'jennifer_lopez', 'hugo_chavez']
if os.environ.get('LORA_IDENTITIES'):
    IDENTITIES = os.environ['LORA_IDENTITIES'].split(',')
VARIANTS = os.environ.get('LORA_VARIANTS', 'clean,mist,mist16,aspl').split(',')
SHARD, SHARDS = map(int, os.environ.get('LORA_SHARD', '0/1').split('/'))
DEV = 'cuda'
RES = 512
PROMPT = 'a photo of sks person'
TRAIN_STEPS = 400
CHECKPOINT_EVERY = 100
RANK = 8
WORK = '/kaggle/working'
OUT = f'{WORK}/lora'
LOG = f'{WORK}/log.txt'
TESTED = f'{WORK}/lora_resume_tested'
STOP_AT_STEP = int(os.environ.get('STOP_AT_STEP', '0'))


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, 'a') as handle:
        handle.write(line + '\n')


def carry_over_previous_output():
    """Copy LoRAs from an attached earlier version's output, finished or partial."""
    for path in glob.glob('/kaggle/input/**/lora/*/*.pt', recursive=True):
        variant, name = path.split('/')[-2:]
        target = f'{OUT}/{variant}/{name}'
        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy(path, target)
            log(f'carried over {variant}/{name}')
    for path in glob.glob('/kaggle/input/**/lora_resume_tested', recursive=True):
        shutil.copy(path, TESTED)


def save(state, path):
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


os.makedirs(OUT, exist_ok=True)
carry_over_previous_output()

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
    """One path per photo number, from the inputs or from kaggle_aspl.py's output."""
    found = {}
    for root in ('/kaggle/input/**', f'{WORK}/protected'):
        for path in glob.glob(f'{root}/{variant}/{identity}/*.png', recursive=True):
            found.setdefault(os.path.basename(path), path)
    return sorted(found.values(), key=lambda p: int(os.path.basename(p)[:-4]))


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


def train_lora(variant, identity, stop_at):
    final = f'{OUT}/{variant}/{identity}.pt'
    partial = f'{OUT}/{variant}/{identity}.partial.pt'
    if os.path.exists(final):
        return 'already done'
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
        log(f'lora {variant}/{identity} resumed at step {start}')
        open(TESTED, 'w').write('resume seen\n')
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
            save({'weights': get_peft_model_state_dict(unet), 'step': step + 1}, partial)
        if stop_at and step + 1 == stop_at:
            log(f'lora {variant}/{identity} stopped at step {stop_at} (resume test)')
            sys.exit(0)
    save({'weights': get_peft_model_state_dict(unet), 'step': TRAIN_STEPS}, final)
    if os.path.exists(partial):
        os.remove(partial)
    del unet, optimizer
    torch.cuda.empty_cache()
    return 'trained'


stop_at = 0 if os.path.exists(TESTED) else STOP_AT_STEP
jobs = [(variant, identity) for variant in VARIANTS for identity in IDENTITIES]
for variant, identity in jobs[SHARD::SHARDS]:
    if len(photos(variant, identity)) < 8:
        log(f'lora {variant}/{identity} skipped: {len(photos(variant, identity))} photos')
        continue
    started = time.time()
    outcome = train_lora(variant, identity, stop_at)
    stop_at = 0
    log(f'lora {variant}/{identity} {outcome} ({time.time() - started:.0f}s)')
log('training done')
