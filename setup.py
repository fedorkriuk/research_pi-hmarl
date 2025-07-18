#\!/usr/bin/env python
"""Setup script for PI-HMARL: Physics-Informed Hierarchical Multi-Agent Reinforcement Learning"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="pi-hmarl",
    version="0.1.0",
    author="PI-HMARL Team",
    author_email="team@pi-hmarl.com",
    description="Physics-Informed Hierarchical Multi-Agent Reinforcement Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pi-hmarl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "ray[rllib]>=2.5.0",
        "gymnasium>=0.28.0",
        "numpy>=1.21.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "wandb>=0.13.0",
        "pybullet>=3.2.0",
        "h5py>=3.7.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pi-hmarl-train=pi_hmarl.train:main",
            "pi-hmarl-eval=pi_hmarl.evaluate:main",
            "pi-hmarl-demo=pi_hmarl.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pi_hmarl": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "data/real_parameters/*.json",
            "data/real_parameters/*.csv",
        ],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pi-hmarl/issues",
        "Source": "https://github.com/yourusername/pi-hmarl",
        "Documentation": "https://pi-hmarl.readthedocs.io",
    },
)
EOF < /dev/null