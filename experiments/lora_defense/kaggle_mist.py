"""Run Mist v2 at 16/255 of [-1, 1] (8/255 of [0, 1]) on Kaggle for more people.

Same tool, environment and command as the ``mist16`` variant of lora_defense_v2.ipynb,
on the photographs under ``/kaggle/input/**/clean/<identity>/`` that have no Mist
output yet. The two T4s of a "GPU T4 x2" session each run one person at a time.

Mist makes five rounds of PGD then LoRA training (about four minutes each on a T4), so
after every round but the last the surrogate's LoRA weights and the perturbed photos
are saved to ``/kaggle/working/mist_resume/<identity>.pt`` and a rerun resumes from
there. Results land in ``/kaggle/working/protected/mist16/<identity>/<n>.png``. An
earlier version's output attached as input is carried over first. ``MIST_STOP_AFTER=n``
stops the first people after round n, once, to test the resume.
"""

import glob
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
WORK = '/kaggle/working'
TEMP = '/kaggle/temp'
LOG = f'{WORK}/log.txt'
CLASS_DIR = f'{TEMP}/class_person'
RESUME_DIR = f'{WORK}/mist_resume'
PROTECTED = f'{WORK}/protected/mist16'
SD_DIR = f'{TEMP}/sd15'
VENV = f'{TEMP}/venv_mist'
REPO = f'{TEMP}/mist-v2'
TESTED = f'{WORK}/mist_resume_tested'
GPUS = 2
# Stop starting people after this long, so the version ends normally and Kaggle keeps
# /kaggle/working (a session killed at the 12-hour limit may keep nothing).
DEADLINE = time.time() + float(os.environ.get('MIST_HOURS', '9')) * 3600


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, 'a') as handle:
        handle.write(line + '\n')


def sh(command, logfile=f'{WORK}/mist_shell.log'):
    with open(logfile, 'a') as handle:
        handle.write(f'$ {command}\n')
        handle.flush()
        done = subprocess.run(command, shell=True, stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode:
        tail = open(logfile).read()[-3000:]
        raise RuntimeError(f'failed ({done.returncode}): {command}\n{tail}')


def identities():
    found = glob.glob('/kaggle/input/**/clean/*/*.png', recursive=True)
    return sorted({path.split('/')[-2] for path in found})


def clean_photos(identity):
    return glob.glob(f'/kaggle/input/**/clean/{identity}/*.png', recursive=True)


def already_protected(identity):
    here = glob.glob(f'{PROTECTED}/{identity}/*.png')
    given = glob.glob(f'/kaggle/input/**/mist16/{identity}/*.png', recursive=True)
    return len(here) >= 8 or len(given) >= 8


def carry_over_previous_output():
    for pattern, target_of in [
            ('/kaggle/input/**/mist_resume/*.pt', lambda p: f'{RESUME_DIR}/{os.path.basename(p)}'),
            ('/kaggle/input/**/protected/mist16/*/*.png',
             lambda p: f'{PROTECTED}/{p.split("/")[-2]}/{os.path.basename(p)}'),
            ('/kaggle/input/**/mist_resume_tested', lambda p: TESTED)]:
        for path in glob.glob(pattern, recursive=True):
            target = target_of(path)
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy(path, target)


def class_images():
    """The 200 prior-preservation images of lora_defense_v2.ipynb: same prompt and seeds."""
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
    """lora_defense_v2.ipynb's venv_mist."""
    if os.path.exists(f'{VENV}/bin/accelerate'):
        return
    os.makedirs(TEMP, exist_ok=True)
    open(f'{TEMP}/build_constraints.txt', 'w').write('setuptools<70\n')
    os.environ['UV_BUILD_CONSTRAINT'] = f'{TEMP}/build_constraints.txt'
    if not os.path.isdir(REPO):
        sh(f'git clone -q https://github.com/psyker-team/mist-v2.git {REPO}')
    sh(f'uv venv -q --python 3.10 {VENV}')
    sh(f'uv pip install -q --python {VENV}/bin/python lit')
    sh(f'uv pip install -q --python {VENV}/bin/python '
       'torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118')
    sh(f'uv pip install -q --python {VENV}/bin/python '
       '"diffusers==0.21.4" "transformers==4.33.3" "accelerate==0.23.0" "huggingface_hub==0.17.3" '
       'ftfy tqdm scipy fire safetensors opencv-python-headless colorama torchmetrics '
       'xformers==0.0.22 "numpy<2" pillow "datasets==2.14.6" "pyarrow<15" pynvml tensorboard Jinja2 '
       'git+https://github.com/cloneofsimo/lora.git "setuptools<70"')


def local_model():
    if os.path.exists(f'{SD_DIR}/model_index.json'):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, local_dir=SD_DIR, allow_patterns=[
        'model_index.json', '*/*.json', '*/*.txt', 'unet/diffusion_pytorch_model.safetensors',
        'vae/diffusion_pytorch_model.safetensors', 'text_encoder/model.safetensors',
        'safety_checker/model.safetensors'])


RESUME = '''    f = [unet, text_encoder]
    start = 0
    resume = os.environ.get('MIST_RESUME')
    if resume and os.path.exists(resume):
        saved = torch.load(resume, map_location='cpu')
        inject_trainable_lora(f[0], r=args.lora_rank)
        for (up, down), (up_state, down_state) in zip(extract_lora_ups_down(f[0]), saved['lora']):
            up.load_state_dict(up_state)
            down.load_state_dict(down_state)
        perturbed_data = saved['perturbed']
        start = saved['done']
        del saved
        print(f"Resumed after round {start} from {resume}", flush=True)
    for i in range(start, args.max_train_steps):
'''

SAVE = '''            print("=======Epoch {} ends!======".format(i))
        if resume and i + 1 < args.max_train_steps:
            lora = [(up.state_dict(), down.state_dict()) for up, down in extract_lora_ups_down(f[0])]
            torch.save({'done': i + 1, 'perturbed': perturbed_data.detach().cpu(), 'lora': lora},
                       resume + '.part')
            os.replace(resume + '.part', resume)
            print(f"Saved round {i + 1} to {resume}", flush=True)
            stop_after = int(os.environ.get('MIST_STOP_AFTER', '0'))
            if stop_after and i + 1 >= stop_after:
                print(f"Stopped after round {i + 1} (resume test)", flush=True)
                raise SystemExit(0)
'''

# Mist checks free memory on GPU 0 whichever GPU it runs on; with two runs side by side
# it has to look at its own.
OWN_GPU = "pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')))"


def patch():
    path = f'{REPO}/attacks/mist.py'
    sh(f'cd {REPO} && git checkout -q attacks/mist.py')
    code = open(path).read()
    loop = '    f = [unet, text_encoder]\n    for i in range(args.max_train_steps):'
    start = code.index(loop)
    end = code.index('\n', start + len(loop)) + 1
    code = code[:start] + RESUME + code[end:]
    for old, new in [
            ('            print("=======Epoch {} ends!======".format(i))\n', SAVE)]:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    assert code.count('pynvml.nvmlDeviceGetHandleByIndex(0)') == 2
    code = code.replace('pynvml.nvmlDeviceGetHandleByIndex(0)', OWN_GPU)
    open(path, 'w').write(code)


def mist(identity, gpu, stop_after):
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
    stop = f'MIST_STOP_AFTER={stop_after} ' if stop_after else ''
    logfile = f'{WORK}/mist_{identity}.log'
    sh(f'cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} MIST_RESUME={resume} {stop}'
       f'PYTHONUNBUFFERED=1 {VENV}/bin/accelerate launch --num_processes 1 attacks/mist.py '
       f'--cuda --low_vram_mode --pretrained_model_name_or_path {SD_DIR} '
       f'--instance_data_dir {instance} --output_dir {work} '
       f'--class_data_dir {CLASS_DIR} --instance_prompt "a photo of sks person" '
       f'--class_prompt "a photo of person" --mixed_precision fp16 '
       f'--pgd_eps {16 / 255}', logfile)
    return sorted(glob.glob(f'{work}/*_noise_*.png')), logfile


def run(identity, gpu, stop_after):
    started = time.time()
    found, logfile = mist(identity, gpu, stop_after)
    if stop_after:
        log(f'mist16 {identity} stopped after round {stop_after} for the resume test')
        return
    if 'Resumed after round' in open(logfile).read():
        open(TESTED, 'w').write('resume seen\n')
    if len(found) < 8:
        raise RuntimeError(f'mist16 {identity}: {len(found)} photos')
    out = f'{PROTECTED}/{identity}'
    os.makedirs(out, exist_ok=True)
    for path in found:
        number = os.path.basename(path).split('_')[-1].split('.')[0]
        shutil.copy(path, f'{out}/{number}.png')
    os.remove(f'{RESUME_DIR}/{identity}.pt')
    log(f'mist16 {identity} done in {time.time() - started:.0f}s on GPU {gpu}')


def worker(gpu, queue, stop_after):
    time.sleep(120 * gpu)  # not both loading the models into host memory at once
    while queue:
        if time.time() > DEADLINE:
            log(f'GPU {gpu}: deadline reached, {len(queue)} left for the next version')
            return
        identity = queue.pop(0)
        run(identity, gpu, stop_after)
        if stop_after:
            return


carry_over_previous_output()
todo = [identity for identity in identities() if not already_protected(identity)]
log(f'mist16 to do ({len(todo)}): {todo}')
if todo:
    environment()
    local_model()
    class_images()
    log(f'class images: {len(glob.glob(CLASS_DIR + "/*.png"))}')
    patch()
    stop_after = 0 if os.path.exists(TESTED) else int(os.environ.get('MIST_STOP_AFTER', '0'))
    with ThreadPoolExecutor(GPUS) as pool:
        for future in [pool.submit(worker, gpu, todo, stop_after) for gpu in range(GPUS)]:
            future.result()
    if stop_after:
        sys.exit(0)
log('mist done')
