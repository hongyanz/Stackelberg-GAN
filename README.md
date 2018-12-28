# Stackelberg-GAN
This is the code for the paper "Stackelberg GAN: Towards Provable Minimax Equilibrium via Multi-Generator Architectures".

The code is written in python and requires matplotlib, torch, torchvision and the tqdm library.

## Architecture of Stackelberg GAN
<p align="center">
    <img src="images/archtecture.pdf" width="800"\>
</p>

## Experimental Results

### Mixture of Gaussians

We test the performance of varying architectures of GANs on a synthetic mixture of Gaussians dataset with 8 modes and 0.01 standard deviation. We observe the following phenomena:

Naïvely increasing capacity of one-generator architecture does not alleviate mode collapse. It shows
that the multi-generator architecture in the Stackelberg GAN effectively alleviates the mode collapse issue.
Though naïvely increasing capacity of one-generator architecture alleviates mode dropping issue, for more
challenging mode collapse issue, the effect is not obvious

#### Running Example
<p align="center">
    <img src="example.png" width="800"\>
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
