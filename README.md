# Embracing Multi-Label Speech Emotion Recognition Using the All-inclusive Aggregation Rule
double-blind submission for INTERSPEECH 2024

# Requirement
* Python ==3.10
* Conda
* Pytorch
* HuggingFace
## Setup environments
* Step0: Install Conda
* Step1:
  ``` bash
  $ conda create -n all_inclusive python=3.10 -y
  $ conda activate all_inclusive
  $ conda install pip
  ```
* Step2: Install Pytorch [Link](https://pytorch.org/get-started/locally/)
* Step3: Install HuggingFace [Link](https://huggingface.co/docs/transformers/installation)

## Change config.json
- Open config.json
```json
{
    "wav_dir": "directory that contains all wav files of MSP-Podcast v1.11",
    "label_path": "file path of labels_consensus.csv for MSP-Podcast v1.11"
}
```
## Train SER systems
``` bash
## Train FT baseline
$ python train_ft_dim_ser.py \
    --seed=0 \
    --ssl_type=wavlm-large \
    --batch_size=32 \
    --accumulation_steps=4 \
    --lr=1e-5 \
    --epochs=10 \
    --model_path=${model_path}
```
## Evaluate baseline model
``` bash
$ python eval_dim_ser.py \
    --ssl_type=wavlm-large \
    --model_path=${model_path}
```
* Run **bash run.sh** to automatically train and evaluate the baseline models
 

