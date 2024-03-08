# Embracing Multi-Label Speech Emotion Recognition Using the All-inclusive Aggregation Rule
double-blind submission for INTERSPEECH 2024

# Requirement
* Python ==3.10
* Conda ==23.11.0
* Pytorch ==2.20 
* HuggingFace ==4.36.2

# Dataset Preparation
* Download WAV files into the folder for each database (e.g., data/MSP-PODCAST1.11/Audios) by submitting the EULA form for the six databases.
  * [USC-IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)
  * [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)
  * [MSP-PODCAST1.11](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
  * [BIIC-PODCAST1.01](https://biic.ee.nthu.edu.tw/open_resource_detail.php?id=63)
  
## Setup environments
* Step0: Install Conda [Link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
* Step1:
  ``` bash
  (base) $ conda create -n all_inclusive python=3.10 -y
  (base) $ conda activate all_inclusive
  (all_inclusive) $ conda install pip
  ```
* Step2: Install Pytorch [Link](https://pytorch.org/get-started/locally/)
* Step3: Install HuggingFace [Link](https://huggingface.co/docs/transformers/installation)


# Train SER systems
* Run **bash run_all_{database_nanme}.sh** to automatically train and evaluate the baseline models
* For instance:
  ``` bash
  (all_inclusive) $ bash run_all_BIIC_PODCAST_P.sh
  (all_inclusive) $ bash run_all_IEMOCAP.sh
  (all_inclusive) $ bash run_all_MSP_IMPROV_P.sh
  (all_inclusive) $ bash run_all_PODCAST_P.sh
  ```

# Evaluate SER systems

* Run **bash run_all_{database_nanme}_Evaluation.sh** to automatically train and evaluate the baseline models
* For instance:
  ``` bash
  (all_inclusive) $ bash run_all_BIIC_PODCAST_P_Evaluation.sh
  (all_inclusive) $ bash run_all_IEMOCAP_Evaluation.sh
  (all_inclusive) $ bash run_all_MSP_IMPROV_P_Evaluation.sh
  (all_inclusive) $ bash run_all_PODCAST_P_Evaluation.sh
  ```

# Check Results
* All **check point** files will be in the folder, **model**
  
