
from setuptools import setup, find_packages

setup(
    name="rltoolbox_wrappers",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "rltoolbox",
        "gymnasium",
        "torch",
    ],
)
