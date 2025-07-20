#!/usr/bin/env python3
"""
Setup script for RLtoolbox: A highly configurable reinforcement learning framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A highly configurable reinforcement learning framework"

# Read version from package
version = "0.1.0"

setup(
    name="rltoolbox",
    version=version,
    author="RLtoolbox Contributors",
    author_email="",
    description="A highly configurable reinforcement learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RLtoolbox",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "gymnasium>=0.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "examples": [
            "matplotlib",
            "seaborn",
            "jupyter",
        ],
    },
    entry_points={
        "console_scripts": [
            "rltoolbox-train=rltoolbox.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
