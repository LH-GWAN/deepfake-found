"""Run Mist v2 on Kaggle for more people, alone or together with the swap shield.

Same tool, environment and command as the ``mist16`` variant of lora_defense_v2.ipynb,
at 16/255 of [-1, 1] (8/255 of [0, 1]) unless a variant says otherwise. ``MIST_VARIANTS``
picks the variants (comma-separated, default ``mist16``):

- ``mist16``: Mist on the clean photographs (``clean/<identity>/``);
- ``shield_mist16``: Mist on photographs that already carry the watermark and the swap
  shield (``shield512/<identity>/``, from prepare_joint_inputs.py): the shield first,
  Mist after;
- ``joint8``, ``joint12``: Mist on the watermarked photographs (``wm512/<identity>/``)
  with the shield's loss added to its own PGD (joint_arcface.py), in one budget of 8/255
  or 12/255 of [0, 1]. ``MIST_JOINT_TAU`` is the similarity at which the shield's step
  stops (default -0.4), ``MIST_JOINT_STEP`` its size (default Mist's own, 0.005) and
  ``ARCFACE_ONNX`` the encoder's path when it is not an input file itself.

Only the photographs under ``/kaggle/input`` that have no output yet are run.
``MIST_PEOPLE`` names the people (comma-separated) instead of everyone found. The jobs
alternate between variants, and the two T4s of a "GPU T4 x2" session each run one at a
time, so they start on different variants.

Mist makes five rounds of PGD then LoRA training (about four minutes each on a T4), so
after every round but the last the surrogate's LoRA weights and the perturbed photos
are saved to ``/kaggle/working/mist_resume/<variant>/<identity>.pt`` and a rerun resumes
from there. Results land in ``/kaggle/working/protected/<variant>/<identity>/<n>.png``.
An earlier version's output attached as input is carried over first.
``MIST_STOP_AFTER=n`` stops each GPU's first job after round n, once per variant, to
test the resume; the mark that a resume was seen carries ``RESUME_TAG``, so a new run
tests afresh.
"""

import glob
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = '/kaggle/working'
TEMP = '/kaggle/temp'
LOG = f'{WORK}/log.txt'
CLASS_DIR = f'{TEMP}/class_person'
RESUME_DIR = f'{WORK}/mist_resume'
PROTECTED = f'{WORK}/protected'
SD_DIR = f'{TEMP}/sd15'
VENV = f'{TEMP}/venv_mist'
REPO = f'{TEMP}/mist-v2'
GPUS = 2
# Variant: (input photographs, budget in 1/255 of [0, 1], shield loss added to the PGD).
VARIANTS = {
    'mist16': ('clean', 8, False),
    'shield_mist16': ('shield512', 8, False),
    'joint8': ('wm512', 8, True),
    'joint12': ('wm512', 12, True),
}
CHOSEN = os.environ.get('MIST_VARIANTS', 'mist16').split(',')
PEOPLE = [p for p in os.environ.get('MIST_PEOPLE', '').split(',') if p]
JOINT_TAU = os.environ.get('MIST_JOINT_TAU', '-0.4')
JOINT_STEP = os.environ.get('MIST_JOINT_STEP', '0.005')
# Stop starting jobs after this long, so the version ends normally and Kaggle keeps
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


def tested(variant):
    # RESUME_TAG (as in kaggle_train_lora.py) makes a new run test afresh instead of
    # inheriting the mark of an earlier run whose output is attached as input.
    return f"{WORK}/mist_resume_tested_{variant}{os.environ.get('RESUME_TAG', '')}"


def find_input(name):
    found = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if not found:
        raise RuntimeError(f'{name} is not among the inputs')
    return found[0]


def identities(variant):
    found = glob.glob(f'/kaggle/input/**/{VARIANTS[variant][0]}/*/*.png', recursive=True)
    everyone = sorted({path.split('/')[-2] for path in found})
    return [p for p in PEOPLE if p in everyone] if PEOPLE else everyone


def source_photos(variant, identity):
    return glob.glob(f'/kaggle/input/**/{VARIANTS[variant][0]}/{identity}/*.png',
                     recursive=True)


def already_protected(variant, identity):
    here = glob.glob(f'{PROTECTED}/{variant}/{identity}/*.png')
    given = glob.glob(f'/kaggle/input/**/{variant}/{identity}/*.png', recursive=True)
    return len(here) >= 8 or len(given) >= 8


def carry_over_previous_output():
    """Copy the chosen variants' checkpoints, photos and resume tests from an earlier output."""
    for variant in CHOSEN:
        for pattern, target_of in [
                (f'/kaggle/input/**/mist_resume/{variant}/*.pt',
                 lambda p: f'{RESUME_DIR}/{variant}/{os.path.basename(p)}'),
                (f'/kaggle/input/**/protected/{variant}/*/*.png',
                 lambda p: f'{PROTECTED}/{variant}/{"/".join(p.split("/")[-2:])}'),
                (f'/kaggle/input/**/{os.path.basename(tested(variant))}',
                 lambda p: tested(variant))]:
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
    """lora_defense_v2.ipynb's venv_mist, plus onnx for the shield's encoder."""
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
       '"onnx==1.14.1" git+https://github.com/cloneofsimo/lora.git "setuptools<70"')


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

# The shield's loss inside Mist's PGD (joint_arcface.py), only when MIST_JOINT_ONNX is set:
# built from the unperturbed images, one extra signed step before Mist's projection, and a
# line per image and round saying how far the similarity is down.
JOINT = [
    ('from attacks.utils import LatentAttack\n',
     'from attacks.utils import LatentAttack\nJOINT = None\n'),
    ('    original_data = perturbed_data.clone()\n', '''    original_data = perturbed_data.clone()
    global JOINT
    if os.environ.get('MIST_JOINT_ONNX'):
        sys.path.insert(0, os.environ['MIST_JOINT_DIR'])
        from joint_arcface import from_environment
        JOINT = from_environment(args.instance_data_dir, original_data)
'''),
    ('                adv_images = perturbed_image + alpha * perturbed_image.grad.sign()\n',
     '''                adv_images = perturbed_image + alpha * perturbed_image.grad.sign()
                if JOINT is not None:
                    adv_images = adv_images + JOINT.step(perturbed_image.detach(), id)
'''),
    ('        image_list.append(perturbed_image.detach().clone().squeeze(0))\n', '''        if JOINT is not None:
            JOINT.report(id)
        image_list.append(perturbed_image.detach().clone().squeeze(0))
'''),
]

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
            ('            print("=======Epoch {} ends!======".format(i))\n', SAVE)] + JOINT:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    assert code.count('pynvml.nvmlDeviceGetHandleByIndex(0)') == 2
    code = code.replace('pynvml.nvmlDeviceGetHandleByIndex(0)', OWN_GPU)
    open(path, 'w').write(code)


def mist(variant, identity, gpu, stop_after):
    _, budget, joint = VARIANTS[variant]
    instance = f'{TEMP}/input/{variant}/{identity}'
    shutil.rmtree(instance, ignore_errors=True)
    os.makedirs(instance)
    for path in source_photos(variant, identity):
        shutil.copy(path, instance)
    work = f'{TEMP}/work/{variant}/{identity}'
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    os.makedirs(f'{RESUME_DIR}/{variant}', exist_ok=True)
    # Set for every job, never inherited: this script's own MIST_STOP_AFTER would otherwise
    # stop a job that should run through, and the joint term is on only when asked for.
    settings = [f'CUDA_VISIBLE_DEVICES={gpu}', f'MIST_RESUME={RESUME_DIR}/{variant}/{identity}.pt',
                'PYTHONUNBUFFERED=1', f'MIST_STOP_AFTER={stop_after}', 'MIST_JOINT_ONNX=']
    if joint:
        encoder = os.environ.get('ARCFACE_ONNX') or find_input('w600k_r50.onnx')
        settings += [f'MIST_JOINT_ONNX={encoder}', f'MIST_JOINT_DIR={HERE}',
                     f'MIST_JOINT_LANDMARKS={find_input("joint_inputs/landmarks.json")}',
                     f'MIST_JOINT_IDENTITY={identity}', f'MIST_JOINT_TAU={JOINT_TAU}',
                     f'MIST_JOINT_STEP={JOINT_STEP}']
    logfile = f'{WORK}/mist_{variant}_{identity}.log'
    sh(f'cd {REPO} && {" ".join(settings)} '
       f'{VENV}/bin/accelerate launch --num_processes 1 attacks/mist.py '
       f'--cuda --low_vram_mode --pretrained_model_name_or_path {SD_DIR} '
       f'--instance_data_dir {instance} --output_dir {work} '
       f'--class_data_dir {CLASS_DIR} --instance_prompt "a photo of sks person" '
       f'--class_prompt "a photo of person" --mixed_precision fp16 '
       f'--pgd_eps {2 * budget / 255}', logfile)
    return sorted(glob.glob(f'{work}/*_noise_*.png')), logfile


def run(variant, identity, gpu, stop_after):
    started = time.time()
    found, logfile = mist(variant, identity, gpu, stop_after)
    if stop_after:
        log(f'{variant} {identity} stopped after round {stop_after} for the resume test')
        return
    if 'Resumed after round' in open(logfile).read():
        open(tested(variant), 'w').write(f'resume seen: {identity}\n')
    if len(found) < 8:
        raise RuntimeError(f'{variant} {identity}: {len(found)} photos')
    out = f'{PROTECTED}/{variant}/{identity}'
    os.makedirs(out, exist_ok=True)
    for path in found:
        number = os.path.basename(path).split('_')[-1].split('.')[0]
        shutil.copy(path, f'{out}/{number}.png')
    os.remove(f'{RESUME_DIR}/{variant}/{identity}.pt')
    log(f'{variant} {identity} done in {time.time() - started:.0f}s on GPU {gpu}')


def worker(gpu, queue, stop_after):
    time.sleep(120 * gpu)  # not both loading the models into host memory at once
    while queue:
        if time.time() > DEADLINE:
            log(f'GPU {gpu}: deadline reached, {len(queue)} left for the next version')
            return
        variant, identity = queue.pop(0)
        run(variant, identity, gpu, 0 if os.path.exists(tested(variant)) else stop_after)
        if stop_after:
            return


unknown = set(CHOSEN) - set(VARIANTS)
if unknown:
    raise SystemExit(f'unknown variants {sorted(unknown)}')
carry_over_previous_output()
queues = [[(v, p) for p in identities(v) if not already_protected(v, p)] for v in CHOSEN]
todo = [job for jobs in zip(*queues) for job in jobs]
todo += [job for jobs in queues for job in jobs[min(map(len, queues)):]]
log(f'mist to do ({len(todo)}): {todo}')
if todo:
    environment()
    local_model()
    class_images()
    log(f'class images: {len(glob.glob(CLASS_DIR + "/*.png"))}')
    patch()
    stop_after = int(os.environ.get('MIST_STOP_AFTER', '0'))
    with ThreadPoolExecutor(GPUS) as pool:
        for future in [pool.submit(worker, gpu, todo, stop_after) for gpu in range(GPUS)]:
            future.result()
    if stop_after:
        sys.exit(0)
log('mist done')
