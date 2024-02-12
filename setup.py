from setuptools import setup, find_packages

setup(
    name='AutoGCN',
    version='1.0.0',
    author='Felix Tempel',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch==2.0.1',
        'tqdm==4.64.0',
        'numpy==1.23.2',
        'omegaconf==2.2.3',
        'setuptools==60.2.0',
        'matplotlib==3.5.3',
        'thop==0.1.1.post2209072238',
        'pynvml==11.4.1',
        'scipy==1.9.1',
        'tensorboard==2.15.2'
    ],
)