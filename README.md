# Installation

## Create an Anaconda Virtual Environment
```
conda create -n moseq-detectron python=3.6
conda activate moseq-detectron
```
## Install PyTorch ≥ 1.7
Install using https://pytorch.org/ to insure you get the correct versions. Something along the lines of:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Install Detectron2
```
python -m pip install git+https://github.com/facebookresearch/detectron2.git
```

## Install this repo
```
git clone https://github.com/tischfieldlab/moseq2-detectron-extract.git
pip install -e moseq2-detectron-extract
```

# Usage

# Generate a training dataset
```
moseq2-detectron-extract generate-dataset /path/to/session.tar.gz
```

# Train a model
```
moseq2-detectron-extract train /path/to/annotations.json /path/to/model-output
```

# Run inference on data using a pre-trained model
```
moseq2-detectron-extract infer /path/to/model/directory/ /path/to/session.tar.gz
```
