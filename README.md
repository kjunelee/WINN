# Wasserstein Introspective Neural Networks


This repository contains the code for the paper:
<br>
[**Wasserstein Introspective Neural Networks**](http://pages.ucsd.edu/%7Eztu/publication/cvpr18_winn.pdf)
<br>
Kwonjoon Lee, Weijian Xu, Fan Fan, [Zhuowen Tu](http://pages.ucsd.edu/~ztu/)   
CVPR 2018 (**Oral**)
<p align='center'>
  <img src='WINN-algorithm.png' width="400px">
</p>

### Introduction

   We present Wasserstein introspective neural networks (WINN) that are both a generator and a discriminator within a single model. WINN provides a significant improvement over the recent introspective neural networks (INN) method by enhancing INN's generative modeling capability. WINN has three interesting properties: (1) A mathematical connection between the formulation of the INN algorithm and that of Wasserstein generative adversarial networks (WGAN) is made. (2) The explicit adoption of the Wasserstein distance into INN results in a large enhancement to INN, achieving compelling results even with a single classifier -- e.g., providing nearly a 20 times reduction in model size over INN for unsupervised generative modeling. (3) When applied to supervised classification, WINN also gives rise to improved robustness against adversarial examples in terms of the error reduction. In the experiments, we report encouraging results on unsupervised learning problems including texture, face, and object modeling, as well as a supervised classification task against adversarial attacks.
   
### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{lee2018winn,
  title={Wasserstein Introspective Neural Networks},
  author={Lee, Kwonjoon and Xu, Weijian and Fan, Fan and Tu, Zhouwen},
  booktitle={CVPR},
  year={2018}
}
```

### Acknowledgments

This code is based on the implementations of [**INNg**](https://github.com/intermilan/inng) and [**WGAN-GP**](https://github.com/igul222/improved_wgan_training).
