"""Run Anti-DreamBooth ASPL on Kaggle for the identities whose protected photos are missing.

Same tool, settings and T4 patches as lora_defense_v2.ipynb (see its protection cell):
the models go to the GPU in bf16 before the loop, train_one_epoch trains in place,
gradient checkpointing, zero_grad(set_to_none=True), PGD in two-image slices with the
surrogate frozen, no xformers, 8-bit AdamW (plain AdamW ran out of memory on a T4).

Every 5 rounds the surrogate weights and the perturbed photos are saved to
``/kaggle/working/aspl_resume/<identity>.pt`` and a rerun resumes from there. Results
land in ``/kaggle/working/protected/aspl/<identity>/<n>.png``. An earlier version's
output attached as input is carried over first. ``ASPL_STOP_AFTER=n`` stops the first
run after round n, once, to test the resume.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

IDENTITIES = ['tom_hanks', 'jennifer_lopez', 'hugo_chavez']
MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
WORK = '/kaggle/working'
TEMP = '/kaggle/temp'
LOG = f'{WORK}/log.txt'
CLASS_DIR = f'{WORK}/class_person'
RESUME_DIR = f'{WORK}/aspl_resume'
PROTECTED = f'{WORK}/protected/aspl'
SD_DIR = f'{TEMP}/sd15'
VENV = f'{TEMP}/venv_aspl'
REPO = f'{TEMP}/Anti-DreamBooth'
TESTED = f'{WORK}/aspl_resume_tested'


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, 'a') as handle:
        handle.write(line + '\n')


def sh(command, logfile=f'{WORK}/aspl_shell.log'):
    with open(logfile, 'a') as handle:
        handle.write(f'$ {command}\n')
        handle.flush()
        done = subprocess.run(command, shell=True, stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode:
        tail = open(logfile).read()[-3000:]
        raise RuntimeError(f'failed ({done.returncode}): {command}\n{tail}')


def clean_photos(identity):
    return glob.glob(f'/kaggle/input/**/clean/{identity}/*.png', recursive=True)


def already_protected(identity):
    here = glob.glob(f'{PROTECTED}/{identity}/*.png')
    given = glob.glob(f'/kaggle/input/**/aspl/{identity}/*.png', recursive=True)
    return len(here) >= 8 or len(given) >= 8


def carry_over_previous_output():
    for pattern, target_of in [
            ('/kaggle/input/**/aspl_resume/*.pt', lambda p: f'{RESUME_DIR}/{os.path.basename(p)}'),
            ('/kaggle/input/**/class_person/*.png', lambda p: f'{CLASS_DIR}/{os.path.basename(p)}'),
            ('/kaggle/input/**/protected/aspl/*/*.png',
             lambda p: f'{PROTECTED}/{p.split("/")[-2]}/{os.path.basename(p)}'),
            ('/kaggle/input/**/aspl_resume_tested', lambda p: TESTED)]:
        for path in glob.glob(pattern, recursive=True):
            target = target_of(path)
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy(path, target)


def class_images():
    os.makedirs(CLASS_DIR, exist_ok=True)
    have = len(glob.glob(f'{CLASS_DIR}/*.png'))
    if have >= 200:
        return
    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                   safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    for index in range(have, 200):
        image = pipe('a photo of person', num_inference_steps=25,
                     generator=torch.Generator('cuda').manual_seed(index)).images[0]
        image.save(f'{CLASS_DIR}/{index}.png')
    del pipe
    torch.cuda.empty_cache()


def environment():
    if os.path.exists(f'{VENV}/bin/python'):
        return
    os.makedirs(TEMP, exist_ok=True)
    open(f'{TEMP}/build_constraints.txt', 'w').write('setuptools<70\n')
    os.environ['UV_BUILD_CONSTRAINT'] = f'{TEMP}/build_constraints.txt'
    sh(f'git clone -q https://github.com/VinAIResearch/Anti-DreamBooth.git {REPO}')
    sh(f'uv venv -q --python 3.10 {VENV}')
    sh(f'uv pip install -q --python {VENV}/bin/python '
       'torch==1.13.1 torchvision==0.14.1 xformers==0.0.16 nvidia-cusparse-cu11')
    sh(f'uv pip install -q --python {VENV}/bin/python '
       '"diffusers==0.13.1" "transformers==4.26.0" "accelerate==0.16.0" "huggingface_hub==0.13.4" '
       '"datasets==2.10.1" ftfy tqdm tensorboard Jinja2 "numpy<2" bitsandbytes==0.41.1 scipy '
       'safetensors pillow "pyarrow<15" "setuptools<70"')
    nvidia = f'{VENV}/lib/python3.10/site-packages/nvidia'
    if not os.path.exists(f'{nvidia}/cuda_runtime/lib/libcudart.so'):
        os.symlink(f'{nvidia}/cuda_runtime/lib/libcudart.so.11.0',
                   f'{nvidia}/cuda_runtime/lib/libcudart.so')


def local_model():
    if os.path.exists(f'{SD_DIR}/model_index.json'):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, local_dir=SD_DIR, allow_patterns=[
        'model_index.json', '*/*.json', '*/*.txt', 'unet/diffusion_pytorch_model.safetensors',
        'vae/diffusion_pytorch_model.safetensors', 'text_encoder/model.safetensors',
        'safety_checker/model.safetensors'])


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
                stop_after = int(os.environ.get('ASPL_STOP_AFTER', '0'))
                if stop_after and i + 1 >= stop_after:
                    print(f"Stopped after round {i + 1} (resume test)", flush=True)
                    raise SystemExit(0)
'''


def patch():
    path = f'{REPO}/attacks/aspl.py'
    sh(f'cd {REPO} && git checkout -q attacks/aspl.py')
    code = open(path).read()
    for old, new in [
            ('    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])\n',
             '    unet, text_encoder = models[0], models[1]\n'),
            ('    f = [unet, text_encoder]\n',
             '    unet.enable_gradient_checkpointing()\n'
             '    text_encoder.gradient_checkpointing_enable()\n'
             "    f = [unet.to('cuda', dtype=torch.bfloat16), "
             "text_encoder.to('cuda', dtype=torch.bfloat16)]\n"),
            ('torch.optim.AdamW(', '__import__("bitsandbytes").optim.AdamW8bit('),
            ('    for i in range(args.max_train_steps):\n', RESUME),
            ('            print(f"Saved noise at step {i+1} to {save_folder}")\n',
             '            print(f"Saved noise at step {i+1} to {save_folder}")\n' + SAVE)]:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    code = code.replace('.zero_grad()', '.zero_grad(set_to_none=True)')
    start = code.index('        perturbed_images.requires_grad = True\n')
    end = code.index('        loss.backward()\n', start) + len('        loss.backward()\n')
    assert code.count('perturbed_images.grad.sign()') == 1
    code = code[:start] + PGD_SLICES + code[end:]
    code = code.replace('perturbed_images.grad.sign()', 'grad.sign()')
    open(path, 'w').write(code)


def aspl(identity, stop_after):
    instance = f'{TEMP}/input/{identity}'
    shutil.rmtree(instance, ignore_errors=True)
    os.makedirs(instance)
    for path in clean_photos(identity):
        shutil.copy(path, instance)
    work = f'{TEMP}/work/{identity}'
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    os.makedirs(RESUME_DIR, exist_ok=True)
    resume = f'{RESUME_DIR}/{identity}.pt'
    nvidia = f'{VENV}/lib/python3.10/site-packages/nvidia'
    libs = ':'.join(f'{nvidia}/{name}/lib' for name in ('cuda_runtime', 'cublas', 'cusparse'))
    stop = f'ASPL_STOP_AFTER={stop_after} ' if stop_after else ''
    sh(f'cd {REPO} && LD_LIBRARY_PATH={libs}:$LD_LIBRARY_PATH ASPL_RESUME={resume} {stop}'
       f'PYTHONUNBUFFERED=1 {VENV}/bin/accelerate launch --mixed_precision fp16 attacks/aspl.py '
       f'--pretrained_model_name_or_path={SD_DIR} '
       f'--instance_data_dir_for_train={instance} --instance_data_dir_for_adversarial={instance} '
       f'--instance_prompt="a photo of sks person" --class_data_dir={CLASS_DIR} '
       f'--num_class_images=200 --class_prompt="a photo of person" --output_dir={work} '
       f'--center_crop --with_prior_preservation --prior_loss_weight=1.0 --resolution=512 '
       f'--train_text_encoder --train_batch_size=1 --max_train_steps=50 '
       f'--max_f_train_steps=3 --max_adv_train_steps=6 --checkpointing_iterations=5 '
       f'--learning_rate=5e-7 --pgd_alpha=5e-3 --pgd_eps=5e-2 --mixed_precision=fp16',
       f'{WORK}/aspl.log')
    return sorted(glob.glob(f'{work}/noise-ckpt/50/*.png'))


carry_over_previous_output()
todo = [identity for identity in IDENTITIES if not already_protected(identity)]
log(f'aspl to do: {todo}')
if todo:
    environment()
    local_model()
    class_images()
    log(f'class images: {len(glob.glob(CLASS_DIR + "/*.png"))}')
    patch()
    stop_after = 0 if os.path.exists(TESTED) else int(os.environ.get('ASPL_STOP_AFTER', '0'))
    for identity in todo:
        started = time.time()
        found = aspl(identity, stop_after)
        if stop_after:
            log(f'aspl {identity} stopped after round {stop_after} for the resume test')
            sys.exit(0)
        if 'Resumed after round' in open(f'{WORK}/aspl.log').read():
            open(TESTED, 'w').write('resume seen\n')
        if len(found) < 8:
            raise RuntimeError(f'aspl {identity}: {len(found)} photos at round 50')
        out = f'{PROTECTED}/{identity}'
        os.makedirs(out, exist_ok=True)
        for path in found:
            number = os.path.basename(path).split('_')[-1].split('.')[0]
            shutil.copy(path, f'{out}/{number}.png')
        os.remove(f'{RESUME_DIR}/{identity}.pt')
        log(f'aspl {identity} done in {time.time() - started:.0f}s (8-bit Adam)')
log('aspl done')
