# Stackelberg-GAN
This is the code for the paper "Stackelberg GAN: Towards Provable Minimax Equilibrium via Multi-Generator Architectures".

The code is written in python and requires matplotlib, torch, torchvision and the tqdm library.

## Architecture of Stackelberg GAN

Stackelberg GAN is a general framework which can be built on top of all variants of standard GANs. The key idea is to apply multiple generators which team up to play against the discriminator.

<p align="center">
    <img src="Stackelberg GAN/images/architecture.png" width="600"\>
</p>

## Experimental Results

### Mixture of Gaussians

We test the performance of varying architectures of GANs on a synthetic mixture of Gaussians dataset with 8 modes and 0.01 standard deviation. We observe the following phenomena:

*Naïvely increasing capacity of one-generator architecture does not alleviate mode collapse*. It shows
that the multi-generator architecture in the Stackelberg GAN effectively alleviates the mode collapse issue.
Though naïvely increasing capacity of one-generator architecture alleviates mode dropping issue, for more
challenging mode collapse issue, the effect is not obvious.

#### Running Example
<p align="center">
    <img src="Stackelberg GAN/images/exp1.png" width="600"\>
</p>

*Stackelberg GAN outperforms multi-branch models.* We compare performance of multi-branch GAN
and Stackelberg GAN with objective functions:

#### Running Example
<p align="center">
    <img src="Stackelberg GAN/images/exp1.png" width="600"\>
</p>

## Reference
For technical details and full experimental results, see [the paper](https://arxiv.org/abs/1811.08010).
```
@article{Zhang2018stackelberg, 
	author = {Hongyang Zhang and Susu Xu and Jiantao Jiao and Pengtao Xie and Ruslan Salakhutdinov and Eric P. Xing}, 
	title = {Stackelberg GAN: Towards Provable Minimax Equilibrium via Multi-Generator Architectures}, 
	journal={arXiv preprint arXiv:1811.08010},
	year = {2018}
}
