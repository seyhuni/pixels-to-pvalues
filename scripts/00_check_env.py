"""Step 0 - environment check: GPU, library versions, write access, HuggingFace access to gated FMs.

python scripts/00_check_env.py            # quick
python scripts/00_check_env.py --hf       # additionally try to fetch config files of gated FMs
"""
import argparse
import importlib
import platform

import _common  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--hf", action="store_true")
a = ap.parse_args()

print("python", platform.python_version(), "|", platform.platform())
for mod in ["torch", "torchvision", "timm", "numpy", "scipy", "pandas", "sklearn", "skimage", "cv2",
            "statsmodels", "yaml", "huggingface_hub", "monai", "nnunetv2", "transformers"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:16s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {mod:16s} MISSING ({type(e).__name__})")

import torch  # noqa: E402

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GB  cc {p.major}.{p.minor}")
    print("  bf16 supported:", torch.cuda.is_bf16_supported())
    x = torch.randn(2048, 2048, device="cuda")
    print("  matmul ok:", float((x @ x).mean()) is not None)

if a.hf:
    from huggingface_hub import hf_hub_download
    from frpv.models.encoders import FM_REGISTRY
    for key, (src, ident, *_rest) in FM_REGISTRY.items():
        repo = ident.replace("hf-hub:", "")
        if "/" not in repo:
            print(f"  {key:10s} timm built-in ({repo})")
            continue
        try:
            hf_hub_download(repo, "config.json")
            print(f"  {key:10s} OK   ({repo})")
        except Exception as e:
            print(f"  {key:10s} NO ACCESS ({repo}): {type(e).__name__} - accept licence on HF + `huggingface-cli login`")
