#!/usr/bin/env python
"""
Web Interface Pre-Launch Validation
Checks all components are ready before starting the server
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version... ", end="")
    if sys.version_info >= (3, 10):
        print("✅ OK (3.10+)")
        return True
    else:
        print(f"❌ FAIL (Found {sys.version_info.major}.{sys.version_info.minor})")
        return False

def check_directories():
    """Check project structure"""
    print("Checking project structure... ", end="")
    required_dirs = [
        "Step-1/code",
        "step-2-layout",
        "step-3",
    ]
    for d in required_dirs:
        if not Path(d).exists():
            print(f"\n❌ FAIL (Missing {d})")
            return False
    print("✅ OK")
    return True

def check_python_packages():
    """Check required packages"""
    print("Checking Python packages...")
    
    packages = {
        "flask": "Flask (web server)",
        "flask_cors": "Flask-CORS (cross-origin requests)",
        "torch": "PyTorch (ML framework)",
        "PIL": "Pillow (image processing)",
        "diffusers": "Diffusers (image generation)",
        "peft": "PEFT (LoRA support)",
    }
    
    all_ok = True
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - missing")
            all_ok = False
    
    return all_ok

def check_pipeline_modules():
    """Check pipeline module imports"""
    print("Checking pipeline modules...")
    
    sys.path.insert(0, str(Path("Step-1/code")))
    sys.path.insert(0, str(Path("step-2-layout")))
    sys.path.insert(0, str(Path("step-3")))
    
    modules = {
        "model_client": "Scene generation",
        "layout_generator": "Layout generation",
        "compositor": "Image composition",
        "text_extractor": "Text extraction",
    }
    
    all_ok = True
    for mod, name in modules.items():
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - {str(e)[:50]}")
            all_ok = False
    
    return all_ok

def check_web_files():
    """Check web interface files"""
    print("Checking web interface files...")
    
    files = [
        ("app.py", "Flask server"),
        ("templates/index.html", "Web interface"),
        ("static/style.css", "Styling"),
        ("static/app.js", "Frontend logic"),
        ("requirements-web.txt", "Dependencies"),
    ]
    
    all_ok = True
    for f, name in files:
        if Path(f).exists():
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} - missing ({f})")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Check GPU availability"""
    print("Checking GPU... ", end="")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ {device_name} ({memory:.1f}GB)")
            return True
        else:
            print("⚠️  No CUDA device found (CPU only)")
            return True  # Not a hard error
    except Exception as e:
        print(f"⚠️  Could not check GPU: {str(e)[:40]}")
        return True

def check_gemini_key():
    """Check Gemini API key"""
    print("Checking Gemini API key... ", end="")
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        print(f"✅ Set ({key[:10]}...)")
        return True
    else:
        print("❌ NOT SET")
        print("   Set with: $env:GEMINI_API_KEY = 'your-key'")
        return False

def check_flask_import():
    """Test Flask import"""
    print("Testing Flask import... ", end="")
    try:
        from flask import Flask
        print("✅ OK")
        return True
    except ImportError:
        print("❌ FAIL")
        return False

def main():
    print("\n" + "=" * 60)
    print("  🎨 Web Interface Pre-Launch Validation")
    print("=" * 60 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_directories),
        ("Pipeline Modules", check_pipeline_modules),
        ("Web Interface Files", check_web_files),
        ("GPU", check_gpu),
        ("Gemini API Key", check_gemini_key),
        ("Flask", check_flask_import),
        ("Python Packages", check_python_packages),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ ERROR in {name}: {e}")
            results[name] = False
        print()
    
    # Summary
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("\n🚀 Ready to start! Run:")
        print("   python app.py")
        print("\nOr use launch script:")
        print("   .\\start_web.ps1")
        return 0
    elif passed >= total - 2:
        print(f"⚠️  MOSTLY OK ({passed}/{total})")
        print("\nYou can try running, but some issues may occur.")
        return 1
    else:
        print(f"❌ FAILED ({passed}/{total})")
        print("\nFix issues above before running.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
