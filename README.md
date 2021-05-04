# BYO-GAN

As a part of my independent study for Junior year, my final project was to recreate StyleGan in Python using Pytorch. I used some notes from a Coursera course I took, the original whitepapers, and looked at how others had implemented the same architecture in order to better understand how StyleGan worked. 

### Whitepapers:
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
  - [Official Implementation](https://github.com/tkarras/progressive_growing_of_gans)
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
  - [Official Implementation](https://github.com/NVlabs/stylegan)


### Datasets:
- [Anime Profile Dataset](https://www.kaggle.com/prasoonkottarathil/gananime-lite)
  - [License](https://creativecommons.org/licenses/by-sa/4.0/)
- [Abstract Art Dataset](https://www.kaggle.com/bryanb/abstract-art-gallery)
- [Abstract Art Images](https://www.kaggle.com/greg115/abstract-art)
- [FFHQ Dataset](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)

### MISC:
Big thanks to these projects for helping me troubleshoot issues I had!
- [huangzh13](https://github.com/huangzh13/StyleGAN.pytorch)
- [rosinality](https://github.com/rosinality/style-based-gan-pytorch)

# How to run

## Preparing a dataset

Create a folder under `/data`, name it however you want.

Place all of your dataset images into this new folder. It will work best if they are all 512x512.

Run `python prep.py [path to images] [start size (4)] [end size (512)]`. At the moment, this script is ***SUPER*** dodgy and thrown together, so be prepared to tweak it in order to make it work. It essentially moves those original images into a new `/data/[name]/original/images` folder. Then, it resizes every image to match progressive growth and saves it under separate datasets under `/data/[name]/prepared`. 

## Training

Edit the `config.txt` file and create a configuration setting to your liking. Use the two examples as a template. You can override any key, but do **NOT** delete anything under the `DEFAULT` setting. 

```shell
python main.py [config name] -c checkpoint.pth
```

For instance,

```shell
python main.py abstract-art
```

runs with the `abstract-art` configuration.

## Getting Full Resolution Samples

TODO

## TODOS:

- add proper device handling
- learning rate scheduling with config