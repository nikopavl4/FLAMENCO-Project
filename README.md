# FLAMENCO

This repo contains the experiments for the FLAMENCO (TERMINET) project.

FLAMENCO is TERMINET's Open Call Topic 1 Winner Proposal. FLAMENCO applied Federated Learning to an existing software application suite, which is used to diagnose communication skills development in children, detecting potential deficiencies timely and accurately. The software suite collects data from a child's responses to an animation game, along with heart rate readings obtained from a smartwatch. The data is then sent and stored into a cloud data hub and analyzed using AI-based classification techniques. The outcome is a risk indicator that suggests the likelihood of a child developing  any learning and communication disorders. As the data collected was sensitive and personal, and the predictive model requires continuous and incremental training, more sophisticated techniques were required. The FLAMENCO project employed several algorithms to ensure users' privacy and prediction accuracy. First, the project used Fully Homomorphic Encryption during the model aggregation step to protect user data from potential breaches. Second, 3 client selection techniques were utilised to handle data and model heterogeneity and improve the model’s predictive accuracy. Third, 8 state-of-the-art aggregators were introduced to manage data imbalance, ensuring that the Federated Learning model can converge effectively.
The project outcome simulates a Federated Learning process using real-world data from IoT edge devices and incorporating the proposed extensions. The communication between FLAMENCO's hardware modules utilised the MQTT protocol to seamlessly integrate with TERMINET’s existing hardware infrastructure. Additionally, a web application was developed to enhance the overall user experience. Overall, the project’s results demonstrate the potential of this decentralised approach to pave the way towards personalised AI-enabled healthcare solutions  while respecting patient privacy.

## Requirements
The project is dependent on classic data science packages:
- pandas
- matplotlib
- seaborn
- scikit-learn
- numpy
- scipy
- torch
- jupyter notebook
- tenseal
You can install them with pip or/and conda.

Example:
```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install scikit-learn -c conda-forge
conda install -c conda-forge matplotlib
conda install pandas
conda install seaborn -c conda-forge
pip install notebook tenseal
```

## Client Selection
- [X] Random Sampler
- [X] Std Sampler
- [X] Quantity Sampler
- [X] Intelli Sampler

## Aggregation Algorithms 
- [X] SimpleAvg
- [X] FedMedian
- [X] FedAvg
- [X] FedNova
- [X] FedAdagrad
- [X] FedYogi
- [X] FedAdam
- [X] FedAvgM


## Evaluation Metrics 
- [X] Presicion@k
- [X] AUC-ROC
- [X] Average Precision
- [X] SIREOS


## Experiment

You can directly reproduce our experiments for all settings using the following commands for each learning setting.
### Centralized
```commandline
python experiment_public/centralized.py
```

### Local
```commandline
python experiment_public/local.py
```

### Federated
```commandline
python experiment_public/federated.py
```
