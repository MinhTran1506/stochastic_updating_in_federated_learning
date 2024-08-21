# Stochastic Updating in Federated Learning

This repository contains the code and thesis related to the implementation and evaluation of stochastic updating algorithms with dropout masks in a federated learning framework. The project primarily focuses on using the CIFAR-10 dataset and evaluating the effectiveness of these methods in improving model robustness, generalization, and privacy.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Baseline Configuration (No FED)](#running-the-baseline-configuration-no-fed)
  - [Running the Stochastic Updating Configuration (No FED)](#running-the-stochastic-updating-configuration-no-fed)
  - [Running the Baseline Configuration (FED)](#running-the-baseline-configuration-no-fed)
  - [Running the Stochastic Updating Configuration (FED)](#running-the-stochastic-updating-configuration-fed)


## Overview

Federated learning is a decentralized approach to machine learning where multiple clients collaboratively train a model without sharing their data. This project explores the use of stochastic updating with dropout masks to enhance the model’s ability to generalize and maintain robustness, especially in environments with diverse client data.

The main contributions of this project are:
1. Implementation of stochastic updating mechanisms with dropout masks in a federated learning context.
2. Evaluation of the impact of these mechanisms on model performance using the CIFAR-10 dataset.

## Directory Structure

```plaintext
├── code/
│   ├── stochastic_updating/
│   │   ├── cifar10_original.py
│   │   ├── cifar10_original_stochastic.py
│   │   ├── cifar10_fl_original.py
│   │   ├── cifar10_fl_stochastic.py
│   │   ├── net.py
│   │   ├── helper_train.py
│   │   ├── helper_evaluate.py
│   │   └── helper_plotting.py
├── client_api_workspace
│   ├── server
│   ├── site-1
│   │   ├── results
│   └── site-2
│       ├── results
├── README.md
├── results
└── requirements.txt
```

## Installation
1. Clone this repository:
```bash
git clone https://github.com/MinhTran1506/stochastic_updating_in_federated_learning.git
cd stochastic_updating_in_federeated_learning
```
2. Activate a virtual environment (optional but recommended):
```bash
source nvflare_env/bin/activate
```
3. Go to the project folder and install requirements:
```bash
cd examples/hello-world/ml-to-fl/pt
pip install -r requirements.txt
```

## Usage
### Running the Baseline Configuration (No FED)
Run this command to run the baseline using:
```bash
# If you want to train using GPU then run this command:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Then run this
python3 ./code/cifar10_original.py
```
This will execute the baseline training process (without Federated Learning)

You can find the plots of the result in the `./pt/results_original`

### Running the Stochastic Updating Configuration (No FED)
Run this command to run the stochastic model (without Federated Learning)
```bash
# If you want to train using GPU then run this command:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Then run this
python3 ./code/cifar10_original_stochastic.py
```
You can find the plots of the result in the `./pt/results`

### Running the Baseline Configuration (FED)
>If you want to train the Baseline model using Federated Learning, please change the name of the model to `Net` in the `Net.py`

We will use the in-process client API, we choose the sag_pt in_proc job template and run the following command to create the job:
```bash
nvflare job create -force -j ./jobs/client_api -w sag_pt_in_proc -sd ./code/ \
    -f config_fed_client.conf app_script=cifar10_fl_original.py
```
Then we can run it using the NVFlare Simulator:
```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/client_api -w client_api_workspace
```

You can find the plots of the result in the `./pt/client_api_workspace/site-1/results_original` or `./pt/client_api_workspace/site-2/results_original`

### Running the Stochastic Updating Configuration (FED)
>If you want to train the Baseline model using Federated Learning, please change the name of the model to `Net` in the `Net.py`

We will use the in-process client API, we choose the sag_pt in_proc job template and run the following command to create the job:
```bash
nvflare job create -force -j ./jobs/client_api -w sag_pt_in_proc -sd ./code/ \
    -f config_fed_client.conf app_script=cifar10_fl_stochastic.py
```
Then we can run it using the NVFlare Simulator:
```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/client_api -w client_api_workspace
```
You can find the plots of the result in the `./pt/client_api_workspace/site-1/results` or `./pt/client_api_workspace/site-2/results`

> You can modify the `-n 2` and `-t 2` to change the number of nodes and threads in the federated learning simulation


Feel free to reach out if you have any questions or need further assistance with the project.
