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
The main directory denoted as $MAIN for this set of experiments is `./convex_reg`.

### Datasets
```
cd $MAIN/datasets/
```
* **w6a** dataset ```wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w6a```,
* **ijcnn1.bz2** dataset ```wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2```,
* **mushrooms** dataset ```wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms```,
* **a6a** dataset ```wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a```.

### Training and Evaluation
- Please running `run_gd.py` and `run_scafflix.py` to train and evaluate. For each file, dataset from `dataset_name = 'mushrooms' ` can be chosen from the above four datasets. 

- After finish training and testing, the intermediate logs will be automatically saved. Running `plot.py` will generated Figure 1 in the paper. 

- We also provide all saved data under `$MAIN/saved_exp` and all plots under `$MAIN/plots`.

## Nonconvex Neural Network Generalization Experiments
We train and test each experiments by running it on a single A100-80G version. 

Now the main folder is `./nn/src`. Datasets will be automatically downloaded. 

### FEMNIST Dataset

- Scafflix compared with baselines
```
cd $MAIN/

python emnist_main_v2.py --flix --scafflix --fedavg --stat_every 10 --max_rounds 1000 --flix_num_rounds 1000 --bs 2048 --exp_no 0101 --alpha 0.1
```

You are free to choose different batch size, experiment name, and personalization factor alpha. 

All other ablations can be regarded as some slightly modified version of above. E.g.,

- Different alpha only
```
python emnist_main_scafflix_only.py --scafflix --stat_every 10 --max_rounds 1000 --flix_num_rounds 1000 --bs 2048 --exp_no 0401 --alpha 10e-4
```

- Different number of clients per round
```
python emnist_main_scafflix_only.py --scafflix --stat_every 10 --max_rounds 1000 --flix_num_rounds 1000 --bs 2048 --exp_no 0501 --alpha 0.3 --n_clients_per_flix_round 1
```

- Different alpha
```
python emnist_main_scafflix_only.py --scafflix --stat_every 10 --max_rounds 1000 --flix_num_rounds 1000 --bs 1 --exp_no 0501 --alpha 0.3
```

### Shakespeare Dataset
```
cd $MAIN/

python shakespeare_main_v1.py --flix --scafflix --fedavg --stat_every 10 --max_rounds 1000 --flix_num_rounds 1000 --bs 2048 --exp_no 01101 --alpha 0.1
```

### Generate Plots
Please refer to jupyter notebooks under `$MAIN/visualization` for more details. 

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
For code related questions, please please contact [Kai Yi](https://kaiyi.me/). For paper related questions, please contact [Kai Yi](https://kaiyi.me/) and [Laurent Condat](https://lcondat.github.io/).
