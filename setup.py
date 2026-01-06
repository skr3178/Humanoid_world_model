from setuptools import setup, find_packages

setup(
    name="masked-hwm",
    version="0.1.0",
    description="Masked Humanoid World Model with Shared Parameters",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "einops>=0.7.0",
        "accelerate>=0.24.0",
        "transformers>=4.35.0",
        "xformers>=0.0.22",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "lpips>=0.1.4",
        "tqdm>=4.66.0",
    ],
    python_requires=">=3.8",
)
