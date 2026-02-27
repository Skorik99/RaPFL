
# A Byzantine-Robust Algorithm for Federated Learning with Partial Participation

## Table of contents
0. [Running Experiments](docs/experiments.md) -- Run the command to get reproducibility
1. [Quickstart](#-quickstart-guide) -- Follow the instructions and get the result!
2. [C4 notation](docs/C4.md) -- Context Container Component Code scheme.
3. [Federated Method Explaining](docs/method.md) -- Get the basis and write your own method
4. [Config Explaining](docs/config.md) -- See allowed optionalization
5. [Attacks](docs/attacks.md) -- Get the basis and write custom attack

## 🚀 Quickstart Guide
### 📋 Prerequisites
1. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

2. **Download dataset**
```bash
python src/federated_datasets/dataset_download.py --dataset-type cifar100 --download-path .
```
Run `python -h src/federated_datasets/dataset_download.py` for details.

### ⚙️ Experiment Setups

#### 🔄 Standard [Federated Averaging](https://arxiv.org/pdf/1602.05629) on CIFAR-10
```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  training_params.device_ids=[0] \
  > fedavg_cifar10_experiment.txt
```

- `device_ids` controls the GPU number (if there are several GPUs on the machine). You can specify multiple ids, then the training will be evenly distributed across the specified devices.
- `manager.batch_size` client processes will be created (higher performance, more resources required). To forcefully terminate the training, kill any of the processes.
- For more details see [Config Explaining](docs/config.md)

#### 🌪️ RapFL on CIFAR10

```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  federated_method=rap \
  dataset@trust_dataset=cifar10 \
  > rap_cifar10_experiment.txt
```

#### 🦠 Byzantine Attacks 

**Rap with Label Flipping Attack**

```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  federated_method=rap \
  dataset@trust_dataset=cifar10 \
  federated_params.clients_attack_types=label_flip \
  federated_params.prop_attack_clients=0.2 \
  federated_params.attack_scheme=constant \
  federated_params.prop_attack_rounds=1.0
  > rap_cifar10_label_flip.txt
```