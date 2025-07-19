#\!/usr/bin/env python
"""Setup script for PI-HMARL: Physics-Informed Hierarchical Multi-Agent Reinforcement Learning"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="pi-hmarl",
    version="1.0.0",
    author="PI-HMARL Team",
    author_email="pi-hmarl@your-org.com",
    description="Physics-Informed Hierarchical Multi-Agent Reinforcement Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/pi-hmarl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.19.0",
        "gymnasium>=0.26.0",
        "networkx>=2.6",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4",
        "tensorboard>=2.8.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "pyzmq>=22.0.0",
        "cryptography>=3.4.0",
        "pyjwt>=2.3.0",
        "lmdb>=1.2.0",
        "numba>=0.54.0",
    ],
    extras_require={
        "hardware": [
            "pymavlink>=2.4.0",
            "pyserial>=3.5",
            "roslibpy>=1.2.0",
            "aiortc>=1.3.0",
        ],
        "optimization": [
            "tensorrt>=8.0.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.9.0",
            "cupy>=9.0.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "visualization": [
            "plotly>=5.3.0",
            "dash>=2.0.0",
            "seaborn>=0.11.0",
            "bokeh>=2.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pi-hmarl=src.cli:main",
            "pi-hmarl-train=src.scripts.train:main",
            "pi-hmarl-evaluate=src.scripts.evaluate:main",
            "pi-hmarl-benchmark=src.scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
        "pi_hmarl": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "data/*.json",
            "data/*.csv",
        ],
    },
    zip_safe=False,
    project_urls={
        "Bug Tracker": "https://github.com/your-org/pi-hmarl/issues",
        "Documentation": "https://pi-hmarl.readthedocs.io",
        "Source Code": "https://github.com/your-org/pi-hmarl",
    },
)
EOF < /dev/null