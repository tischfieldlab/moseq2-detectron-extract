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
        'albumentations',
        'bottleneck',
        'click',
        'click-option-group',
        'elasticdeform',
        'FyeldGenerator',
        'h5py',
        'imageio',
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'ruamel.yaml',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'tabulate',
        'tifffile',
        'tqdm',
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
