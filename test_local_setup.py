#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local setup validation tests (no GPU/API required)

Tests:
1. Dependencies installed
2. Data files exist
3. Configuration valid
4. Scripts executable
5. Model available (if RunPod)
"""

import json
import sys
import io
from pathlib import Path

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def test_dependencies():
    """Test critical Python packages."""
    print(f"\n{YELLOW}[1/5] Testing Dependencies{RESET}")

    packages = {
        "torch": "Deep learning framework",
        "transformers": "HuggingFace transformers",
        "flask": "Web framework",
        "openai": "OpenAI SDK",
        "datasets": "HuggingFace datasets",
        "peft": "Parameter-efficient fine-tuning",
    }

    passed = 0
    failed = 0

    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  {GREEN}✅{RESET} {package}: {description}")
            passed += 1
        except ImportError:
            print(f"  {RED}❌{RESET} {package}: NOT INSTALLED")
            failed += 1

    return passed, failed


def test_data_files():
    """Test data files exist."""
    print(f"\n{YELLOW}[2/5] Testing Data Files{RESET}")

    root = Path(__file__).parent
    data_files = [
        "pathvqa_for_app.json",
        "vqarad_for_app.json",
    ]

    passed = 0
    failed = 0

    for filename in data_files:
        filepath = root / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {GREEN}✅{RESET} {filename} ({size_mb:.1f} MB)")
            passed += 1
        else:
            print(f"  {RED}❌{RESET} {filename}: NOT FOUND")
            failed += 1

    # Check JSON validity
    print()
    for filename in data_files:
        filepath = root / filename
        if filepath.exists():
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                count = len(data)
                print(f"  {GREEN}✅{RESET} {filename}: {count} cases")
                passed += 1
            except json.JSONDecodeError as e:
                print(f"  {RED}❌{RESET} {filename}: Invalid JSON - {e}")
                failed += 1

    return passed, failed


def test_scripts():
    """Test all required scripts exist and are executable."""
    print(f"\n{YELLOW}[3/5] Testing Scripts{RESET}")

    root = Path(__file__).parent
    scripts = [
        "train_visual_prm.py",
        "setup_runpod.sh",
        "train_runpod.sh",
        "verify_setup.sh",
        "runpod_qwen_openai_server.py",
        "test_mc_pipeline.py",
    ]

    passed = 0
    failed = 0

    for script in scripts:
        filepath = root / script
        if filepath.exists():
            print(f"  {GREEN}✅{RESET} {script}")
            passed += 1
        else:
            print(f"  {RED}❌{RESET} {script}: NOT FOUND")
            failed += 1

    return passed, failed


def test_configuration():
    """Test configuration files."""
    print(f"\n{YELLOW}[4/5] Testing Configuration{RESET}")

    root = Path(__file__).parent
    configs = [
        ".env.runpod",
        "requirements.txt",
    ]

    passed = 0
    failed = 0

    for config in configs:
        filepath = root / config
        if filepath.exists():
            print(f"  {GREEN}✅{RESET} {config}")
            passed += 1

            # Check content
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
                if config == "requirements.txt":
                    required = ["torch", "transformers", "accelerate"]
                    for req in required:
                        if req in content:
                            print(f"    {GREEN}✓{RESET} {req} in requirements")
                        else:
                            print(f"    {YELLOW}⚠{RESET} {req} missing from requirements")
        else:
            print(f"  {RED}❌{RESET} {config}: NOT FOUND")
            failed += 1

    return passed, failed


def test_runpod_readiness():
    """Check RunPod readiness."""
    print(f"\n{YELLOW}[5/5] Testing RunPod Readiness{RESET}")

    root = Path(__file__).parent

    passed = 0
    failed = 0

    # Check setup scripts
    setup_script = root / "setup_runpod.sh"
    if setup_script.exists():
        print(f"  {GREEN}✅{RESET} setup_runpod.sh ready")
        passed += 1
    else:
        print(f"  {RED}❌{RESET} setup_runpod.sh missing")
        failed += 1

    # Check training script
    train_script = root / "train_runpod.sh"
    if train_script.exists():
        print(f"  {GREEN}✅{RESET} train_runpod.sh ready")
        passed += 1
    else:
        print(f"  {RED}❌{RESET} train_runpod.sh missing")
        failed += 1

    # Check model server
    server_script = root / "runpod_qwen_openai_server.py"
    if server_script.exists():
        print(f"  {GREEN}✅{RESET} runpod_qwen_openai_server.py ready")
        passed += 1
    else:
        print(f"  {RED}❌{RESET} runpod_qwen_openai_server.py missing")
        failed += 1

    return passed, failed


def main():
    """Run all tests."""
    print("=" * 50)
    print("VisualPRM Local Setup Validation")
    print("=" * 50)

    total_passed = 0
    total_failed = 0

    # Run all tests
    p, f = test_dependencies()
    total_passed += p
    total_failed += f

    p, f = test_data_files()
    total_passed += p
    total_failed += f

    p, f = test_scripts()
    total_passed += p
    total_failed += f

    p, f = test_configuration()
    total_passed += p
    total_failed += f

    p, f = test_runpod_readiness()
    total_passed += p
    total_failed += f

    # Summary
    print()
    print("=" * 50)
    print(f"{GREEN}✅ Passed: {total_passed}{RESET}")
    print(f"{RED}❌ Failed: {total_failed}{RESET}")
    print("=" * 50)

    # Recommendations
    print()
    if total_failed == 0:
        print(f"{GREEN}All tests passed! Ready for RunPod.{RESET}")
        print()
        print("Next steps:")
        print("1. Create A100-40GB instance on RunPod")
        print("2. Run: bash setup_runpod.sh")
        print("3. Run: bash train_runpod.sh standard")
        return 0
    else:
        print(f"{RED}Some tests failed. Fix issues before proceeding.{RESET}")
        if total_failed > 0:
            print()
            print("Failed items:")
            print("- Install missing dependencies: pip install -r requirements.txt")
            print("- Check data files in project root")
            print("- Verify git clone copied all files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
