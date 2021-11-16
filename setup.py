import subprocess
import sys

from setuptools import find_packages, setup


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2  # noqa: F401
except ImportError:
    install('opencv-python')


setup(
    name='moseq2-detectron-extract',
    author='Tischfield Lab',
    description='Network for extracting raw moseq depth data',
    version='0.1.0',
    license='MIT License',
    install_requires=[
        'click',
        'h5py',
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'ruamel.yaml',
        'scipy',
        'scikit-learn',
        'scikit-image',
        'tifffile',
        'imageio',
        'tabulate',
        'tqdm',
        'albumentations',
        'FyeldGenerator',
        'bottleneck',
        'statsmodels'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov'
        ]
    },
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': ['moseq2-detectron-extract = moseq2_detectron_extract.cli:cli'],
    }
)
