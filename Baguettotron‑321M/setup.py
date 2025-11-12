"""
Setup configuration for Baguettotron package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ]

setup(
    name="baguettotron",
    version="0.1.0",
    author="Jacques Gariépy",
    author_email="",
    description="Baguettotron-321M: Independent reverse engineering and from-scratch implementation (educational). Not affiliated with PleIAs. From-scratch implementation of the 321M parameter language model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M",
    project_urls={
        "Bug Tracker": "https://github.com/JacquesGariepy/AI-From-Scratch/issues",
        "Documentation": "https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M",
        "Source Code": "https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M",
        "Official Model": "https://huggingface.co/PleIAs/Baguettotron",
        "SYNTH Dataset": "https://huggingface.co/datasets/PleIAs/SYNTH",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "train": [
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
        "all": [
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "baguettotron-train=scripts.train:main",
            "baguettotron-generate=scripts.generate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
        "language model",
        "natural language processing",
        "nlp",
        "pytorch",
        "baguettotron",
        "llm",
    ],
)
