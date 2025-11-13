"""
Setup configuration for Baguettotron package with intelligent dataset management.

Features:
- Dataset-specific extras (synth, wikipedia, demo)
- Auto-download on install
- Multi-dataset support
- Intelligent dataset detection

Examples:
    # Install with training + Wikipedia dataset
    pip install -e ".[train,wikipedia]"

    # Install with training + multiple datasets
    pip install -e ".[train,synth,wikipedia]"

    # Just download datasets without training
    pip install -e ".[dataset,wikipedia]"

    # Full installation with all datasets
    pip install -e ".[all,synth,wikipedia]"
"""

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from pathlib import Path
import subprocess
import sys


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


class PostInstallCommand:
    """Base class for post-install dataset downloads."""

    @staticmethod
    def download_datasets():
        """Download datasets based on installed extras."""
        try:
            # Import here to avoid circular dependency
            from baguettotron.data.auto_download import download_requested_datasets
            download_requested_datasets()
        except Exception as e:
            print(f"⚠️  Dataset auto-download failed: {e}")
            print("   You can manually download datasets using:")
            print("   python scripts/setup_dataset.py")


class PostDevelopCommand(develop, PostInstallCommand):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        # Auto-download is handled by extras


class PostInstallInstallCommand(install, PostInstallCommand):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Auto-download is handled by extras


# Define extras_require with dataset options
extras_require = {
    # Development
    "dev": [
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
        "black>=23.3.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.3.0",
    ],

    # Training dependencies
    "train": [
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tensorboard>=2.13.0",
        "wandb>=0.15.0",
    ],

    # Dataset-only installation (no training)
    "dataset": [
        "datasets>=2.12.0",
        "transformers>=4.30.0",  # For tokenizer
    ],

    # Individual datasets
    "synth": [
        "datasets>=2.12.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.14.0",
    ],

    "wikipedia": [
        "datasets>=2.12.0",
        "transformers>=4.30.0",
    ],

    "demo": [
        # Demo has no external dependencies (pure Python)
    ],

    # Combined options
    "all-datasets": [
        "datasets>=2.12.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.14.0",
    ],
}

# Add convenience aliases
extras_require["all"] = (
    extras_require["train"] +
    extras_require["dev"] +
    extras_require["all-datasets"]
)

setup(
    name="baguettotron",
    version="1.0.0",
    author="Jacques Gariépy",
    author_email="",
    description="Baguettotron-321M: 321M parameter language model with intelligent dataset management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M",
    project_urls={
        "Bug Tracker": "https://github.com/JacquesGariepy/AI-From-Scratch/issues",
        "Documentation": "https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M",
        "Source Code": "https://github.com/JacquesGariepy/AI-From-Scratch/tree/main/Baguettotron-321M",
        "Official Model": "https://huggingface.co/PleIAs/Baguettotron",
        "SYNTH Dataset": "https://huggingface.co/datasets/PleIAs/SYNTH",
        "Wikipedia Dataset": "https://huggingface.co/datasets/wikipedia",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
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
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "baguettotron=baguettotron.cli:main",
        ],
    },
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallInstallCommand,
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
        "datasets",
        "multi-dataset",
    ],
)
