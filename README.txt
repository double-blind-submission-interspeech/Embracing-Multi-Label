Docker or Conda/Anaconda:

    We provide a docker image in anonymous123321/ar-ser:1.0, which is identical to the envionment we used for the paper. We also provide a file with all the packages used that can be used with conda/Anaconda ("AR_FOR_SER/AR_ENV.txt")

    For Docker please follow the following instruction:
        1. Make sure Docker and CUDA/nvidia-container-runetime are install on machine
        2. Pull the provided Docker image using. NOTE: the Docker image is large and might take some time to download.
	    >docker pull anonymous123321/ar-ser:1.0
        3. Go to the projects root dir ("AR_FOR_SER")
        4. Build the local Docker image, copying the files
           >docker build --tag ar_ser .
        4. Build and run the interactive Docker image using: 
	    >docker run -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all ar_ser:latest /bin/bash
	5. Once in the interactive shell, the project root dir is located in /AR_SER

    For Conda please following the following instructions:
        ## Requirements        
        - Ubuntu 18.04 LTS        
        - Anaconda3        
        - CUDA 11.0        
        - cudnn 8.0.5        
                
        ## Create the Conda environment
        >conda create --name ar_env --file AR_ENV.txt        
        >source activate ar_env        
        

preprocessing:
    CREMA-D:
        1. Make sure to copy the CREMA-D audios to AR_FOR_SER/data/CREMA-D/audios/*.wav
    MSP-IMPROV:
        1. Download the dataset files and use the VIDEO_to_WAV_IMPROV.py code to down-sample and extract audio .wav files from the videos. Make sure the extracted audios are AR_FOR_SER/data/MSP-IMPROV/audios/*.wav
        2. VIDEO_to_WAV_IMPROV.py explanation
		2-1. Write the directory path of MSP-IMPROV corpus in variable name "root" (line 6 in VIDEO_to_WAV_IMPROV.py)
		2-2. run "python VIDEO_to_WAV_IMPROV.py"
		2-3. Once the pre-processing is done, extracted audios will be in AR_FOR_SER/data/MSP-IMPROV/audios/*.wav

    MSP-PODCAST:
        1. Make sure to copy the MSP-PODCAST audios to AR_FOR_SER/data/MSP-PODCAST1.10/audios/*.wav
    USC-IMOCAP:
        1. Make sure to copy the USC-IMOCAP audios to AR_FOR_SER/data/USC-IEMOCAP/audios/*.wav
        
        
        

Train/test:
    All the script files used for the paper are located in AR_FOR_SER/run/<dataset>/*.sh
    To run any of the files, copy it to the AR_FOR_SER dir and run in AR_FOR_SER: 
    >bash run/dataset/example.sh 
    
        
    
