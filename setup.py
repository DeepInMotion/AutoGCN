from setuptools import setup, find_packages

setup(
    name='AutoGCN',
    version='1.0.0',
    author='Felix Tempel',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch~=1.13.0.dev20220826',
        'tqdm',
        'tensorboard',
        'PyYAML',
        'omegaconf~=2.2.3',
        'setuptools~=60.2.0',
        'torchvision~=0.14.0.dev20220825',
        'matplotlib~=3.5.3',
        'thop',
        'pynvml',
        'tensorboardX',
        'numpy',
        'scipy',
    ],
)