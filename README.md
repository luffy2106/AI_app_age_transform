

# For Development

### Install conda
1. Download and Install anaconda in /tmp 
```
wget -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
```
2. Activate conda
```
. anaconda3/bin/activate
```
3. Verify install:
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
conda activate age_venv
```

### Download model
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


# Install Nvidia Container Toolkit
This step is necessary to connect your GPU to docker

https://github.com/NVIDIA/nvidia-docker

sudo nvidia-ctk runtime configure

Test NVIDIA docker:
sudo docker run --rm --gpus all nvidia/cuda:10.1-base nvidia-smi

Other reference :
https://gitlab.com/dataScienceAssystem/aramco/-/tree/E-cataloging-v2-final?ref_type=heads

# Run with docker
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