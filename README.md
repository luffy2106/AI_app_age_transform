

# For Development(Only apply for Linux or WSL2 window), skip it you want to deploy the project
### 1. Install cuda to enable GPU in WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

Checking:
If you see cuda folder in usr/local, and you can see cuda version by the following command, then you are success:
```
nvidia-smi
```

### 2. Install conda
Download and Install anaconda(mini version) 
```
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /tmp/conda 
```
Activate conda
```
. /tmp/conda/bin/activate
```
Verify install:
```
conda --version
```
If you see conda version, then you are sucessful

### 3. Install virtual environment
Install
```
conda env create -f environment/kien_env_37.yaml
```
Activate virtual environment
```
conda activate age_venv_37
```
### 4. Download model
Create folder to store model:
```
mkdir pretrained_models
```
Download age transform model:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC" -O ./pretrained_models/sam_ffhq_aging.pt && rm -rf /tmp/cookies.txt
```
### 5. Download face detector lib
This is dlib(deep learning library) of python to recognize the facial points of any person's face.
```
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
```

### 6. Install package to run notebook (optional, please skip it if you are doing deployment)
If you want to add virutal environment to jupyter
```
source activate myenv
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```
If you want to see logs of jupyter in vscode : 
- ctlr + shift + P => jupyter : show output


### 7. Install package to build dlib(choose one option)
7.1. Using ninja-linux(faster than CMAKE)
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip 
&& sudo apt install unzip
&& sudo unzip ninja-linux.zip -d /usr/local/bin/
&& sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```

Note that some old OS system does not support ninja-linux. In that case, install cmake instead.

7.2. Using cmake 
```
sudo apt-get install cmake
```
### 8. Running
You can change the age number as you want, note that the age number should be integer
```
python scripts/inference.py --age 40
```
You could see the result in folder <mark>output</mark>
### 9. Deploy to fastAPI
```
uvicorn scripts.main:app --host 0.0.0.0 --reload --port 6080
```

# For Deployment
## 1. Install linux OS system

Install Ubuntu 18.04 or 20.04 or \

Windows Subsystem for Linux(WSL) if you are using window. See the tutorial in this <a href="https://learn.microsoft.com/en-us/windows/wsl/install" target="_blank">link</a>

Remember that Docker only support WSL 2, so you should make sure that you installed WSL 2, you can check you are on which version by:
```
wsl -l -v
```

## 2. Install docker
Follow the tutorial in this link :
```
https://docs.docker.com/engine/install/ubuntu/
```

## 3. Enable GPU on docker(in case your service use GPU)
Docker doesn’t even add GPUs to containers by default so a plain docker run won’t see your hardware at all. You need to configure it
- Make sure you’ve got the NVIDIA drivers working properly on your host before you continue with your Docker configuration. You should be able to successfully run nvidia-smi and see your GPU’s name, driver version, and CUDA version
- Adding the NVIDIA Container Toolkit to your host:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - 
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
- Install the nvidia-docker2 package on your host:
```
apt-get update
apt-get install -y nvidia-docker2
```
- Test if gpu enable sucessfully in docker:
```
docker run -it --gpus all nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi
```
You shoud see your GPU’s name, driver version, and CUDA version

Reference:
```
https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/
```

## 4. Run with docker
Build the image
```
docker build -t age_transform -f docker/Dockerfile .
```
Build and run by docker compose
```
docker-compose up -d
```

# Few notes on this project(For development only)
- There are some dependencies in files YAML which can not convert to requirements.txt, because there dependencices managed by conda only
- You can run the tool by unicorn, but if you "-k uvicorn.workers.UvicornWorker" and "--timeout 180" the program will show error, because uvicorn need to set up the process to run the tool and the default timeout of the command is short than we expected.
```
gunicorn scripts.main:app -w 1 --preload --timeout 180 -k uvicorn.workers.UvicornWorker -b "0.0.0.0:6080" 
```
- If you bind port in docker compose file and set up command in docker compose, no need to set up in dockerfile anymore. 

