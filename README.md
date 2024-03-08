# Embracing Multi-Label Speech Emotion Recognition Using the All-inclusive Aggregation Rule
double-blind submission for INTERSPEECH 2024

# Requirement
* Python ==3.10
* Conda ==23.11.0
* Pytorch ==2.20 
* HuggingFace ==4.36.2
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
## Train and Evaluate SER systems
* 
* Run **bash run.sh** to automatically train and evaluate the baseline models
 

