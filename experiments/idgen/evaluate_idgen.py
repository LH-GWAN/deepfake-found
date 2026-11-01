r"""Does the swap shield also stop single-photo identity generators?

Tools such as IP-Adapter FaceID and InstantID draw a person from one photograph with no
training: they read the face with an ArcFace model and condition a diffusion model on
that embedding. IP-Adapter FaceID reads it with ``buffalo_l`` (the very model the shield
attacks), InstantID with ``antelopev2`` (another ArcFace). This feeds each tool the clean
photograph, the shielded one (``protect --mode shield`` as shipped, watermark included)
and the Mist + shield one from the anti-LoRA test, generates, and asks whether the
result shows the person, among the whole gallery, with the pipeline's ArcFace and SFace.

Tools:
``faceid``    IP-Adapter FaceID on Stable Diffusion 1.5 (ArcFace embedding only);
``plusv2``    IP-Adapter FaceID Plus v2 (ArcFace embedding plus a CLIP view of the face);
``instantid`` InstantID on SDXL (antelopev2 embedding plus face-landmark ControlNet).

Images are written as they are made and a rerun skips them. Runs locally on MPS:
generation only, nothing is trained.

    python experiments/idgen/evaluate_idgen.py --tool faceid --refs 4
    python experiments/idgen/evaluate_idgen.py --tool instantid --refs 2
    python experiments/idgen/evaluate_idgen.py --score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

V3 = ROOT / "experiments/lora_defense/v3"
CONDITIONS = {
    "clean": V3 / "clean",
    "shield": V3 / "protected/shield",
    "mist16_shield": V3 / "protected/mist16_shield",
}
# The attacker knows whom they are copying; with a gender-neutral prompt SD 1.5 often drew a
# woman for a man even from the clean photograph.
PROMPT = "a photo of a {gender}, portrait, natural light, high quality"
NEGATIVE = "blurry, low quality, deformed, cartoon, drawing"
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
INSIGHTFACE_ROOT = ROOT / "models/insightface"


def people() -> list[str]:
    return sorted(p.name for p in (V3 / "protected/mist16_shield").iterdir() if p.is_dir())


def face_app(name: str):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=name, root=str(INSIGHTFACE_ROOT), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def largest_face(app, rgb: np.ndarray):
    faces = app.get(rgb[:, :, ::-1].copy())  # insightface wants BGR
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def faceid_pipeline(plus: bool):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
    from transformers import CLIPVisionModelWithProjection

    kwargs = {}
    if plus:
        kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    # Only the float32 VAE is cached; SD 1.5's VAE runs fine in float16.
    kwargs["vae"] = AutoencoderKL.from_pretrained(SD15, subfolder="vae", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15, variant="fp16", torch_dtype=torch.float16, safety_checker=None, **kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    weight = "ip-adapter-faceid-plusv2_sd15.bin" if plus else "ip-adapter-faceid_sd15.bin"
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name=weight,
                         image_encoder_folder=None)
    pipe.set_ip_adapter_scale(1.0)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_faceid(pipe, app, rgb: np.ndarray, plus: bool, seed: int, gender: str):
    import torch
    from insightface.utils import face_align

    face = largest_face(app, rgb)
    if face is None:
        return None
    # (batch, images per prompt, tokens, dim), as in diffusers' IP-Adapter FaceID example.
    embed = torch.from_numpy(face.normed_embedding).reshape(1, 1, 1, 512)
    id_embeds = torch.cat([torch.zeros_like(embed), embed]).to("mps", torch.float16)
    if plus:
        crop = face_align.norm_crop(rgb, landmark=face.kps, image_size=224)
        clip = pipe.prepare_ip_adapter_image_embeds(
            [Image.fromarray(crop)], None, torch.device("mps"), 1, True)[0]
        layer = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        layer.clip_embeds = clip.to(torch.float16)
        layer.shortcut = True  # Plus v2
    return pipe(PROMPT.format(gender=gender), negative_prompt=NEGATIVE,
                ip_adapter_image_embeds=[id_embeds],
                num_inference_steps=30, guidance_scale=7.5, width=512, height=512,
                generator=torch.Generator("cpu").manual_seed(seed)).images[0]


def instantid_pipeline():
    import torch
    from diffusers import ControlNetModel

    sys.path.insert(0, str(HERE))
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
    from huggingface_hub import snapshot_download

    instantid = Path(snapshot_download("InstantX/InstantID",
                                       allow_patterns=["ControlNetModel/*", "ip-adapter.bin"]))
    controlnet = ControlNetModel.from_pretrained(instantid / "ControlNetModel",
                                                 torch_dtype=torch.float16)
    # From the cached folder itself: by repository name diffusers also wants vae_1_0, which
    # was not downloaded and is not used.
    from huggingface_hub import try_to_load_from_cache

    sdxl = Path(try_to_load_from_cache(SDXL, "model_index.json")).parent
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        sdxl, controlnet=controlnet, variant="fp16", torch_dtype=torch.float16)
    pipe.to("mps")
    pipe.load_ip_adapter_instantid(str(instantid / "ip-adapter.bin"))
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_instantid(pipe, app, rgb: np.ndarray, seed: int, gender: str):
    import torch
    from pipeline_stable_diffusion_xl_instantid import draw_kps

    face = largest_face(app, rgb)
    if face is None:
        return None
    # The repository's resize_img: long side to 1280 then down to a multiple of 64; our
    # photographs are small squares, so 1024 x 1024 with the landmarks scaled along.
    scale = 1024 / rgb.shape[0]
    photo = Image.fromarray(rgb).resize((1024, 1024), Image.BICUBIC)
    kps = draw_kps(photo, face.kps * scale)
    return pipe(PROMPT.format(gender=gender), negative_prompt=NEGATIVE,
                image_embeds=face.embedding, image=kps,
                controlnet_conditioning_scale=0.8, ip_adapter_scale=0.8,
                num_inference_steps=30, guidance_scale=5.0,
                generator=torch.Generator("cpu").manual_seed(seed)).images[0]


def run(tool: str, refs: int, out: Path, limit: int | None = None) -> None:
    plus = tool == "plusv2"
    if tool == "instantid":
        pipe, app = instantid_pipeline(), face_app("antelopev2")
    else:
        pipe, app = faceid_pipeline(plus), face_app("buffalo_l")
    # The attacker knows whom they are copying, so the prompt names their gender; it is read
    # from a file because the pack's gender estimate is often wrong on these photographs.
    genders = json.loads((V3 / "genders.json").read_text())
    report = out / tool / "no_face.json"
    no_face: dict[str, list[str]] = json.loads(report.read_text()) if report.exists() else {}
    for person in people()[:limit]:
        for n in range(refs):
            for condition, folder in CONDITIONS.items():
                target = out / tool / condition / person / f"{n}.png"
                key = f"{condition}/{person}/{n}"
                if target.exists() or key in no_face.get(condition, []):
                    continue
                rgb = np.asarray(Image.open(folder / person / f"{n}.png").convert("RGB"))
                image = (generate_instantid(pipe, app, rgb, n, genders[person])
                         if tool == "instantid"
                         else generate_faceid(pipe, app, rgb, plus, n, genders[person]))
                if image is None:
                    no_face.setdefault(condition, []).append(key)
                    report.parent.mkdir(parents=True, exist_ok=True)
                    report.write_text(json.dumps(no_face, indent=1))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target)
        print(f"{tool} {person} done", flush=True)


def score(out: Path, gallery: Path, output: Path) -> None:
    from evaluate_source_protection import Recognisers
    from evaluate_source_protection import score as rank

    from deepshield.media import load_image

    recognise = Recognisers()
    grouped: dict[str, list[Path]] = {}
    for path in sorted(gallery.glob("*.png")):
        grouped.setdefault(path.stem.rsplit("_", 1)[0], []).append(path)
    galleries = {i: {p.name: v for p in ps if (v := recognise(load_image(p))) is not None}
                 for i, ps in grouped.items() if len(ps) >= 3}
    report: dict = {"question": "does the swap shield stop single-photo identity generators?",
                    "gallery_identities": len(galleries),
                    "arcface_high_confidence": recognise.high, "tools": {}}
    for tool_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        no_face_path = tool_dir / "no_face.json"
        no_face = json.loads(no_face_path.read_text()) if no_face_path.exists() else {}
        tool: dict = {}
        for condition in CONDITIONS:
            entries = []
            for path in sorted((tool_dir / condition).glob("*/*.png")):
                person, n = path.parent.name, path.stem
                entries.append(rank(recognise(load_image(path)), galleries, person,
                                    f"{person}_{n}.png"))
            scored = [e for e in entries if e is not None]
            sims = [e["arcface"]["donor_similarity"] for e in scored]
            tool[condition] = {
                "generated": len(entries),
                "reference_face_not_found": len(no_face.get(condition, [])),
                "with_a_face": len(scored),
                "arcface_person_first": int(sum(e["arcface"]["donor_first"] for e in scored)),
                "arcface_median_similarity": round(float(np.median(sims)), 4) if sims else None,
                "arcface_above_high_confidence": int(sum(v >= recognise.high for v in sims)),
                "sface_person_first": int(sum(e["sface"]["donor_first"] for e in scored)),
                "sface_median_similarity": (
                    round(float(np.median([e["sface"]["donor_similarity"] for e in scored])), 4)
                    if scored else None),
            }
        report["tools"][tool_dir.name] = tool
    output.mkdir(parents=True, exist_ok=True)
    (output / "idgen.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=["faceid", "plusv2", "instantid"])
    parser.add_argument("--refs", type=int, default=4, help="reference photographs per person")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--people", type=int, default=None, help="only the first N, for a trial")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--gallery", type=Path, default=V3 / "gallery")
    parser.add_argument("--output", type=Path, default=ROOT / "data/results/idgen")
    args = parser.parse_args()
    if args.score:
        score(args.out, args.gallery, args.output)
    else:
        run(args.tool, args.refs, args.out, args.people)


if __name__ == "__main__":
    main()
