# Install conda
Download and Install anaconda in /tmp 
- cd /tmp
- wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
- bash Anaconda3-2023.03-1-Linux-x86_64.sh 
- . anaconda3/bin/activate
Verify install:
- conda --version

# Install virtual environment
Install
- conda env create -f environment/kien_env_37.yaml
Activate
- conda activate age_venv
Install jinja2 to run template on fastAPI
- pip install jinja2
Install multipart to take form data request
- pip install python-multipart

# Download model

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC" -O ./pretrained_models/sam_ffhq_aging.pt && rm -rf /tmp/cookies.txt

# Install package to run notebook 

If you want to add virutal environment to jupyter
- source activate myenv
- pip install ipykernel
- python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
If you want to see logs of jupyter in vscode : 
- ctlr + shift + P => jupyter : show output

# Install face detector 
This is dlib(deep learning library) of python to recognize the facial points of any person's face.
- wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"

# Package to build dlib(choose one option)
1. Using ninja-linux(faster than CMAKE)

- wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
- sudo apt install unzip
- sudo unzip ninja-linux.zip -d /usr/local/bin/
- sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 

Note that some old OS system does not support ninja-linux. In that case, install cmake instead.

2. Using cmake 
- sudo apt-get install cmake

# Running
You can change the age number as you want, note that the age number should be integer
- python scripts/inference.py --age 40

# Deploy to fastAPI
gunicorn scripts.main:app -w 1 --timeout 180 -k uvicorn.workers.UvicornWorker -b "0.0.0.0:8000"


# Run with docker
1. Build the image

docker build -t age_transform -f docker/Dockerfile .

2. Run the container

docker run -d --gpus all age_transform:latest



Reference :

https://pythonspeed.com/articles/activate-conda-dockerfile/