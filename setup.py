from setuptools import setup, find_packages

setup(
    name="slotcontrast",
    version="0.1.0",
    description="slotcontrast/setup.py",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    # install_requires=[
    #     # NOTE: torch and torchvision should be installed separately via conda
    #     # conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    #     # Do not include torch/torchvision here to avoid overriding conda installation
    #     "numpy>=1.24.0",
    #     "einops>=0.6.0",
    #     "omegaconf>=2.3.0",
    #     "hydra-core>=1.3.0",
    #     "pytorch-lightning>=2.0.0",
    #     "tensorboard>=2.13.0",
    #     "matplotlib>=3.7.0",
    #     "opencv-python>=4.8.0",
    #     "pillow>=10.0.0",
    #     "tqdm>=4.65.0",
    #     "pyyaml>=6.0.0",
    #     "scipy>=1.10.0",
    #     "torchmetrics>=1.0.0",
    # ],
    # extras_require={
    #     "dev": [
    #         "pytest>=7.4.0",
    #         "black>=23.7.0",
    #         "isort>=5.12.0",
    #     ],
    #     "viz": [
    #         "viser>=0.1.0",
    #         "imageio>=2.31.0",
    #     ],
    # },
)
