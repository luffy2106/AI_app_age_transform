

# For Development(Only apply for Linux or WSL2 window)
### Install cuda to enable GPU in WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

Cheking:
If you see cuda folder in usr/local, and you can see cuda version by the following command, then you are success:
```
nvidia-smi
```

### Install conda
1. Download and Install anaconda 
1.1 Mini version
```
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p conda 
```

1.2 Activate conda
```
. ~/conda/bin/activate
```
1.3 Verify install:
```
conda --version
```

### Install virtual environment
1. Install
```
conda env create -f environment/kien_env_37.yaml
```
2. Activate virtual environment
```
conda activate age_venv_37
```

### Download model
Create folder to store model:
```
mkdir pretrained_models
```

Download age transform model:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC" -O ./pretrained_models/sam_ffhq_aging.pt && rm -rf /tmp/cookies.txt
```

### Download face detector lib
This is dlib(deep learning library) of python to recognize the facial points of any person's face.
```
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
```

### Install package to run notebook (optional)
If you want to add virutal environment to jupyter
- source activate myenv
- pip install ipykernel
- python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
If you want to see logs of jupyter in vscode : 
- ctlr + shift + P => jupyter : show output


### Install package to build dlib(choose one option)
1. Using ninja-linux(faster than CMAKE)
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip 
&& sudo apt install unzip
&& sudo unzip ninja-linux.zip -d /usr/local/bin/
&& sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```

Note that some old OS system does not support ninja-linux. In that case, install cmake instead.

2. Using cmake 
```
sudo apt-get install cmake
```
### Running
You can change the age number as you want, note that the age number should be integer
```
python scripts/inference.py --age 40
```
### Deploy to fastAPI
```
gunicorn scripts.main:app -w 1 --timeout 180 -k uvicorn.workers.UvicornWorker -b "0.0.0.0:6080"
```

uvicorn scripts.main:app -w 1 --timeout 180 -k uvicorn.workers.UvicornWorker -b "0.0.0.0:6080"


uvicorn scripts.main:app --host 0.0.0.0 --port 7000
# For Deployment
## 3.0 Enable GPU on docker(in case your service use GPU)
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

## Run with docker
1. Build the image

docker build -t age_transform -f docker/Dockerfile .

2. Run the container

docker run -d --gpus all age_transform:latest

docker run -p 6090:6080 -d --gpus all age_transform:latest

uvicorn scripts.main:app --host 0.0.0.0 --port 7000

3. Build docker image and run the container in the background

docker-compose up -d

Reference :

https://pythonspeed.com/articles/activate-conda-dockerfile/