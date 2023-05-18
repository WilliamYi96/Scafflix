# Explicit Personalization and Local Training: Double Communication Acceleration in Federated Learning

This repository contains the code to run all experiments presented in our paper [Explicit Personalization and Local Training: Double Communication Acceleration in Federated Learning](link).

## Overview
Federated Learning is an evolving machine learning paradigm, in which multiple clients perform computations based on their individual private data, interspersed by communication with a remote server. A common strategy to curtail communication costs is Local Training, which consists in performing multiple local stochastic gradient descent steps between successive communication rounds. However, the conventional approach to local training overlooks the practical necessity for client specific personalization, a technique to tailor local models to individual needs. We introduce Scafflix, a novel algorithm that efficiently integrates explicit person9 alization with local training. This innovative approach benefits from these two techniques, thereby achieving doubly accelerated communication, as we demon11 strate both in theory and practice.

## Environment Setup
```angular2html
# create a conda virtual environment
conda create --name scafflix python=3.6
# activate the conda environment
conda activate scafflix
# check https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier to install fedjax
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Convex Logistic Regression Experiments

## Nonconvex Neural Network Generalization Experiments

## Citation
```
@misc{scafflix,
  author = {Kai Yi, Laurent Condat, Peter Richtarik},
  title = {Explicit Personalization and Local Training: Double Communication Acceleration in Federated Learning},
  year = {2023},
  journal={arXiv preprint},
}
```

## Acknowledgement
We would like to extend our special appreciation to [FedJax](https://github.com/google/fedjax) for their exceptional implementation of FedAvg and for the incorporation of the datasets API. Furthermore, we are grateful to [FLIX](https://github.com/google/fedjax) for generously sharing their valuable FLIX implementation.

## License

The intended purpose and licensing of Scafflix is solely for research use.

The source code is licensed under Apache 2.0.

## Contact
For more information please contact [Kai Yi](https://kaiyi.me/), [Laurent Condat](https://lcondat.github.io/).
