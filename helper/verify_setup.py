#!/usr/bin/env python3
"""
Comprehensive setup verification script for Humanoid World Model.

This script verifies that all components are correctly installed and configured:
- CUDA and GPU availability
- PyTorch installation
- Cosmos Tokenizer setup
- Dataset availability
- Model imports
- Checkpoint structure
- Basic forward pass test
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_check(name: str, status: bool, details: str = ""):
    """Print a check result with status."""
    status_symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"

    print(f"{color}{status_symbol} {name:<40} [{status_text}]{reset}")
    if details:
        print(f"  {details}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    is_valid = version.major == 3 and version.minor >= 8
    return is_valid, version_str


def check_cuda_availability() -> Tuple[bool, str]:
    """Check CUDA availability and version."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            details = f"CUDA {cuda_version}, {device_count} GPU(s), {device_name}"
        else:
            details = "CUDA not available"
        return cuda_available, details
    except ImportError:
        return False, "PyTorch not installed"


def check_gpu_memory() -> Tuple[bool, str]:
    """Check GPU memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "No GPU available"

        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / 1e9
        torch.cuda.empty_cache()

        # Get current memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        free = total_memory_gb - reserved

        details = f"Total: {total_memory_gb:.2f}GB, Free: {free:.2f}GB, Used: {reserved:.2f}GB"
        is_sufficient = total_memory_gb >= 20  # At least 20GB recommended

        return is_sufficient, details
    except Exception as e:
        return False, str(e)


def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch installation."""
    try:
        import torch
        version = torch.__version__
        return True, f"v{version}"
    except ImportError:
        return False, "Not installed"


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"v{version}"
    except ImportError:
        return False, "Not installed"


def check_cosmos_tokenizer() -> Tuple[bool, str]:
    """Check Cosmos Tokenizer installation and model files."""
    try:
        from cosmos_tokenizer import DiscreteTokenizer, ContinuousTokenizer
        return True, "Package installed"
    except ImportError as e:
        return False, f"Import failed: {e}"


def check_cosmos_models(base_dir: Path) -> Tuple[bool, str]:
    """Check Cosmos Tokenizer model files."""
    tokenizer_dir = base_dir / "cosmos_tokenizer"

    required_files = ["encoder.jit", "decoder.jit", "config.json"]
    found_files = []
    missing_files = []

    for filename in required_files:
        file_path = tokenizer_dir / filename
        if file_path.exists():
            found_files.append(filename)
        else:
            missing_files.append(filename)

    if missing_files:
        return False, f"Missing: {', '.join(missing_files)}"
    else:
        return True, f"Found: {', '.join(found_files)}"


def check_dataset(base_dir: Path) -> Tuple[bool, str]:
    """Check dataset availability."""
    data_dir = base_dir / "1xgpt" / "data"

    # Check for v2.0 dataset structure
    train_dir = data_dir / "train_v2.0"
    val_dir = data_dir / "val_v2.0"
    test_dir = data_dir / "test_v2.0"

    results = []
    all_exist = True

    if train_dir.exists():
        num_shards = len(list(train_dir.glob("*.tar")))
        results.append(f"Train: {num_shards} shards")
    else:
        results.append("Train: MISSING")
        all_exist = False

    if val_dir.exists():
        num_shards = len(list(val_dir.glob("*.tar")))
        results.append(f"Val: {num_shards} shards")
    else:
        results.append("Val: MISSING")
        all_exist = False

    if test_dir.exists():
        results.append("Test: OK")
    else:
        results.append("Test: MISSING (optional)")

    return all_exist, ", ".join(results)


def check_build_folder(base_dir: Path) -> Tuple[bool, str]:
    """Check build folder structure."""
    build_dir = base_dir / "build"

    if not build_dir.exists():
        return False, "Build directory not found"

    required_items = [
        "masked_hwm",
        "training",
        "data",
        "train_full_v2_rtx4090.sh",
    ]

    found = []
    missing = []

    for item in required_items:
        if (build_dir / item).exists():
            found.append(item)
        else:
            missing.append(item)

    if missing:
        return False, f"Missing: {', '.join(missing)}"
    else:
        return True, f"All components present ({len(found)} items)"


def check_model_import(base_dir: Path) -> Tuple[bool, str]:
    """Check if model can be imported."""
    try:
        # Add build directory to path
        build_dir = base_dir / "build"
        sys.path.insert(0, str(build_dir))

        from masked_hwm.config import MaskedHWMConfig, MaskedHWMRTX4090Config
        from masked_hwm.model import MaskedHWM

        return True, "All model modules imported successfully"
    except ImportError as e:
        return False, f"Import failed: {e}"


def check_forward_pass(base_dir: Path) -> Tuple[bool, str]:
    """Test a simple forward pass."""
    try:
        import torch
        build_dir = base_dir / "build"
        sys.path.insert(0, str(build_dir))

        from masked_hwm.config import MaskedHWMRTX4090Config
        from masked_hwm.model import MaskedHWM

        # Create config
        config = MaskedHWMRTX4090Config()

        # Create model (on CPU for quick test)
        model = MaskedHWM(config)
        model.eval()

        # Create dummy input
        batch_size = 2
        num_frames = config.num_past_frames + config.num_future_frames
        spatial_size = config.spatial_size

        # Factorized tokens: (batch, frames, spatial, spatial, num_factors)
        video_tokens = torch.randint(
            0, config.vocab_size,
            (batch_size, num_frames, spatial_size, spatial_size, config.num_factored_vocabs)
        )

        # Actions: (batch, num_past_frames, action_dim)
        actions = torch.randn(batch_size, config.num_past_frames, config.action_dim)

        # Forward pass
        with torch.no_grad():
            outputs = model(video_tokens, actions)

        return True, f"Forward pass successful (output shape: {outputs.shape})"
    except Exception as e:
        return False, f"Forward pass failed: {e}"


def main():
    """Run all verification checks."""
    print_header("Humanoid World Model - Setup Verification")

    # Get base directory
    base_dir = Path(__file__).parent.absolute()
    print(f"\nBase directory: {base_dir}")

    all_checks_passed = True

    # System checks
    print_header("System Checks")

    status, details = check_python_version()
    print_check("Python Version (>= 3.8)", status, details)
    all_checks_passed &= status

    status, details = check_cuda_availability()
    print_check("CUDA Availability", status, details)
    all_checks_passed &= status

    status, details = check_gpu_memory()
    print_check("GPU Memory (>= 20GB recommended)", status, details)
    if not status:
        print("  WARNING: Limited GPU memory may cause training issues")

    # PyTorch and dependencies
    print_header("Core Dependencies")

    status, details = check_pytorch()
    print_check("PyTorch", status, details)
    all_checks_passed &= status

    packages = [
        ("torchvision", "torchvision"),
        ("xformers", "xformers"),
        ("accelerate", "accelerate"),
        ("transformers", "transformers"),
        ("einops", "einops"),
        ("wandb", "wandb"),
    ]

    for pkg_name, import_name in packages:
        status, details = check_package(pkg_name, import_name)
        print_check(pkg_name, status, details)
        all_checks_passed &= status

    # Cosmos Tokenizer
    print_header("Cosmos Tokenizer")

    status, details = check_cosmos_tokenizer()
    print_check("Cosmos Tokenizer Package", status, details)
    all_checks_passed &= status

    status, details = check_cosmos_models(base_dir)
    print_check("Cosmos Tokenizer Models", status, details)
    all_checks_passed &= status

    # Dataset
    print_header("Dataset")

    status, details = check_dataset(base_dir)
    print_check("1x-technologies/worldmodel v2.0", status, details)
    all_checks_passed &= status

    # Build folder
    print_header("Build Folder")

    status, details = check_build_folder(base_dir)
    print_check("Build Folder Structure", status, details)
    all_checks_passed &= status

    # Model import
    print_header("Model Verification")

    status, details = check_model_import(base_dir)
    print_check("Model Import", status, details)
    all_checks_passed &= status

    if status:
        # Only run forward pass if import succeeded
        status, details = check_forward_pass(base_dir)
        print_check("Forward Pass Test", status, details)
        all_checks_passed &= status

    # Final summary
    print_header("Summary")

    if all_checks_passed:
        print("\n✓ All checks passed! Setup is complete and ready for training.")
        print("\nNext steps:")
        print("  1. Activate conda environment: conda activate cosmos-tokenizer")
        print(f"  2. Navigate to build folder: cd {base_dir}/build")
        print("  3. Start training: ./train_full_v2_rtx4090.sh")
        print("\nFor help, see: README_SERVER_SETUP.md")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("  - Ensure conda environment is activated: conda activate cosmos-tokenizer")
        print("  - Check that all dependencies are installed: pip install -r requirements.txt")
        print("  - Verify dataset download completed: check 1xgpt/data/")
        print("  - See README_SERVER_SETUP.md for detailed setup instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
