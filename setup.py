"""
Setup script for Physics-Informed Hierarchical Multi-Agent Reinforcement Learning (PI-HMARL)

This setup script defines the package installation configuration including
dependencies, entry points, and metadata.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


# Get the long description from the README file
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Package metadata
NAME = "pi-hmarl"
DESCRIPTION = "Physics-Informed Hierarchical Multi-Agent Reinforcement Learning"
LONG_DESCRIPTION = long_description
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/yourusername/pi-hmarl"
AUTHOR = "PI-HMARL Team"
AUTHOR_EMAIL = "team@pi-hmarl.com"
LICENSE = "MIT"
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
KEYWORDS = [
    "reinforcement learning",
    "multi-agent systems",
    "physics-informed learning",
    "hierarchical learning",
    "deep learning",
    "machine learning",
    "artificial intelligence",
    "physics simulation",
    "robotics",
]

# Version
def get_version():
    """Get version from __init__.py file."""
    version_file = here / "src" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip("\"'")
    return "0.1.0"

VERSION = get_version()

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Core dependencies
INSTALL_REQUIRES = [
    # Core ML and RL dependencies
    "torch>=2.0.0",
    "ray[rllib]>=2.5.0",
    "gymnasium>=0.28.0",
    "numpy>=1.21.0",
    "scipy>=1.8.0",
    
    # Visualization and plotting
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.0.0",
    
    # Experiment tracking and hyperparameter optimization
    "wandb>=0.13.0",
    "optuna>=3.0.0",
    "tensorboard>=2.10.0",
    
    # Data handling
    "h5py>=3.7.0",
    "pandas>=1.4.0",
    "pickle5>=0.0.11",
    
    # Physics simulation environments
    "pybullet>=3.2.0",
    "mujoco>=2.3.0",
    "dm-control>=1.0.0",
    
    # Configuration and utilities
    "PyYAML>=6.0.0",
    "hydra-core>=1.2.0",
    "omegaconf>=2.2.0",
    "click>=8.0.0",
    
    # Additional utilities
    "tqdm>=4.64.0",
    "psutil>=5.9.0",
    "colorlog>=6.7.0",
    "tabulate>=0.9.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.950",
        "pre-commit>=2.20.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
    ],
    "gpu": [
        "nvidia-ml-py>=11.450.51",
        "pynvml>=11.0.0",
    ],
    "physics": [
        "dm-control>=1.0.0",
        "mujoco-py>=2.1.0",
        "pybullet>=3.2.0",
        "pymunk>=6.4.0",
    ],
    "viz": [
        "pygame>=2.1.0",
        "opencv-python>=4.5.0",
        "imageio>=2.20.0",
        "ffmpeg-python>=0.2.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autodoc-typehints>=1.19.0",
        "myst-parser>=0.18.0",
    ],
    "deployment": [
        "docker>=6.0.0",
        "kubernetes>=25.0.0",
        "mlflow>=2.0.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
    ],
}

# Add 'all' extra that includes all optional dependencies
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Entry points for command-line tools
ENTRY_POINTS = {
    "console_scripts": [
        "pi-hmarl=src.main:main",
        "pi-hmarl-train=src.training.train:main",
        "pi-hmarl-eval=src.evaluation.evaluate:main",
        "pi-hmarl-config=src.utils.config_manager:main",
        "pi-hmarl-setup=scripts.setup_env:main",
    ],
}

# Package data
PACKAGE_DATA = {
    "pi_hmarl": [
        "configs/*.yaml",
        "configs/*.yml",
        "data/*.json",
        "data/*.csv",
        "models/*.pt",
        "models/*.pth",
    ],
}

# Include additional files
INCLUDE_PACKAGE_DATA = True

# Data files (installed outside of package)
DATA_FILES = [
    ("configs", ["configs/default_config.yaml"]),
    ("scripts", ["scripts/setup_env.sh"]),
    ("docker", ["Dockerfile", "docker-compose.yml"]),
]


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    
    def run(self):
        develop.run(self)
        self.execute(self._post_install, [], msg="Running post-install tasks")
    
    def _post_install(self):
        """Execute post-installation tasks."""
        print("Setting up development environment...")
        
        # Create necessary directories
        directories = [
            "logs", "data/raw", "data/processed", "data/results",
            "experiments", "checkpoints", "models"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Set up pre-commit hooks if available
        try:
            import subprocess
            subprocess.run(["pre-commit", "install"], check=True)
            print("Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Pre-commit not available, skipping hooks setup")
        
        print("Development environment setup complete!")


class PostInstallCommand(install):
    """Post-installation for install mode."""
    
    def run(self):
        install.run(self)
        self.execute(self._post_install, [], msg="Running post-install tasks")
    
    def _post_install(self):
        """Execute post-installation tasks."""
        print("Setting up PI-HMARL...")
        
        # Create necessary directories
        directories = [
            "logs", "data", "experiments", "checkpoints", "models"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        print("PI-HMARL installation complete!")


# Custom commands
CMDCLASS = {
    "develop": PostDevelopCommand,
    "install": PostInstallCommand,
}


def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 8):
        raise RuntimeError(
            "Python 3.8 or higher is required. "
            f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )


def get_install_requires():
    """Get install requirements, handling potential import errors."""
    try:
        return INSTALL_REQUIRES
    except Exception as e:
        print(f"Warning: Error reading requirements: {e}")
        return []


def main():
    """Main setup function."""
    check_python_version()
    
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        url=URL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        classifiers=CLASSIFIERS,
        keywords=" ".join(KEYWORDS),
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        python_requires=PYTHON_REQUIRES,
        install_requires=get_install_requires(),
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        package_data=PACKAGE_DATA,
        include_package_data=INCLUDE_PACKAGE_DATA,
        data_files=DATA_FILES,
        cmdclass=CMDCLASS,
        zip_safe=False,
        
        # Additional metadata
        project_urls={
            "Bug Reports": f"{URL}/issues",
            "Source": URL,
            "Documentation": f"{URL}/docs",
        },
        
        # Options for bdist_wheel
        options={
            "bdist_wheel": {
                "universal": False,
            },
        },
    )


if __name__ == "__main__":
    main()