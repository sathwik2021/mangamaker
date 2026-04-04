import sys, traceback
try:
    import torch
    print(f"torch OK: {torch.__version__}")
except Exception as e:
    print(f"torch FAIL: {e}")
    sys.exit(1)

try:
    from diffusers import StableDiffusionPipeline
    print("diffusers OK")
except Exception as e:
    print(f"diffusers FAIL: {e}")
    traceback.print_exc()

try:
    from peft import PeftModel
    print("peft OK")
except Exception as e:
    print(f"peft FAIL: {e}")
    traceback.print_exc()

try:
    from PIL import Image
    print("PIL OK")
except Exception as e:
    print(f"PIL FAIL: {e}")

try:
    import numpy
    print("numpy OK")
except Exception as e:
    print(f"numpy FAIL: {e}")
