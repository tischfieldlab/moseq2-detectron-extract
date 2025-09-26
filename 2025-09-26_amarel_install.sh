# below are commands for installing detectron2 and moseq2-detectron-extract on amarel

# load a more recent gcc version
module purge
module load gcc/10.2.0-bz186


# create the environment
conda create -n moseq-detectron python=3.8


# activate the newly-created environment
conda activate moseq-detectron


# install ffmpeg. need to specify openh264 version
# othewise ffmpeg is broken (having been built with a different version than conda will give you)
# also need to use channel_priority flexible (temp), and reset to strict after install
conda config --set channel_priority flexible
conda install -c pytorch ffmpeg openh264=2.1.0
conda config --set channel_priority strict


# install pytorch
conda install pytorch=1.10.0 torchvision=0.11.0 torchaudio=0.10.0 cudatoolkit=11.3.1 -c pytorch


# downgrade mkl to 2024.0.0, otherwise detectron2 fails to build
conda install mkl=2024.0.0


# install detectron2
python -m pip install git+https://github.com/facebookresearch/detectron2.git@58e472e076a5d861fdcf773d9254a3664e045bf8


# clone and install moseq2-detectron-extract
git clone https://github.com/tischfieldlab/moseq2-detectron-extract.git
pip install -e moseq2-detectron-extract/


# uninstall and reinstall pillow to get pypi version (conda version seems to be broken)
pip uninstall pillow
pip install pillow==9.4.0

# test installation
moseq-d2-extract --help

