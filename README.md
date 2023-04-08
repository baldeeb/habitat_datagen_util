# Rough README

## Setup

Install Mamba

```
mamba create -n habitat python=3.8 cmake=3.14.0
mamba activate habitat
<!-- mamba install habitat-sim withbullet -c conda-forge -c aihabitat -->
mamba install habitat-sim headless -c conda-forge -c aihabitat
mamba install -c conda-forge wandb quaternion

<!-- mamba install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -->
<!-- mamba install -c pytorch faiss-cpu faiss-gpu -->

pip install distance-map bresenham python-opencv  magnum git+https://github.com/lucasb-eyer/pydensecrf.git

git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
rm -rf habitat-lab/
```


## References
* [habitat sim](https://github.com/facebookresearch/habitat-sim)
* [habitat lab](https://github.com/facebookresearch/habitat-lab)
Based off of [One-Shot Transfer of Affordance Regions? AffCorrs!](https://sites.google.com/view/affcorrs) which uses [DINO-ViT]()

## Dependencies

Use python 3.7

The dino bits of the code require the following libraries. This is copied from others and is not revised

```python
pydensecrf=1.0
torch=1.10+cu113
faiss=1.5
pytorch_metric_learning=0.9.99
fast_pytorch_kmeans=0.1.6
timm=0.6.7
cv2=4.6
scikit-image=0.17.2
```

# [Habitat](https://aihabitat.org/)

## Installation:

- Simulator: ```conda install habitat-sim withbullet -c conda-forge -c aihabitat```
- Laboratory: [detailed here](https://github.com/facebookresearch/habitat-lab)

## [Data Downloading](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md):
Use the data_downloader.py script from the util section of the repo
