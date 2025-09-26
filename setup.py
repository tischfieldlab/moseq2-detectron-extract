import subprocess
import sys

from setuptools import find_packages, setup


def install(package: str):
    ''' install a `package` via pip
    '''
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2  # pylint: disable=unused-import
except ImportError:
    install('opencv-python')


setup(
    name='moseq2-detectron-extract',
    author='Tischfield Lab',
    description='Network for extracting raw moseq depth data',
    version='0.1.0',
    license='MIT License',
    install_requires=[
        'albumentations==1.1.0',
        'bottleneck==1.3.6',
        'click==8.1.3',
        'click-option-group==0.5.5',
        'elasticdeform==0.5.0',
        'FyeldGenerator==0.1.7',
        'h5py==3.8.0',
        'imageio==2.25.0',
        'joblib==1.2.0',
        'matplotlib==3.6.3',
        'norfair==2.2.0',
        'numpy==1.24.1',
        'opencv-python==4.7.0.68',
        'opencv-python-headless==4.7.0.68',
        'pandas==1.5.3',
        'pillow==9.4.0',
        'protobuf~=3.20.3',
        'pykalman==0.9.5',
        'ruamel.yaml==0.17.21',
        'scikit-image==0.19.3',
        'scikit-learn==1.2.1',
        'scipy==1.10.0',
        'statsmodels==0.13.5',
        'tabulate==0.9.0',
        'tifffile==2023.1.23.1',
        'tqdm==4.64.1',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
            'mypy'
        ]
    },
    python_requires='>=3.8',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'moseq2-detectron-extract = moseq2_detectron_extract.cli:cli',
            'moseq-d2-extract = moseq2_detectron_extract.cli:cli' # add short alias
        ],
    }
)
