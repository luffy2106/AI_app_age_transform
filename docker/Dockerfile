#FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
# # Set the image name label
# LABEL name="age_transform"


# Set the timezone environment variable
ENV TZ=Europe/London


# Set the DEBIAN_FRONTEND environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive


# Install necessary dependencies
RUN    apt-get update \
    && apt-get install -y software-properties-common \
    && apt-get install wget \
    && apt-get update \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get -y install python3-pip \
    && apt-get install -y libgl1-mesa-dev \
    && apt-get install -y cmake \
    && apt-get install python3.7-dev \
    # && apt-get install -y ninja-build \
    && apt install unzip \
    && apt-get clean

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip 
RUN unzip ninja-linux.zip -d /usr/local/bin/ 
RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 


# Set the environment variables for Conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy files to docker image
COPY . /app/age_transform_api
WORKDIR /app/age_transform_api


# Install package
RUN pip install -r requirements.txt


# Create a new Conda environment
# RUN conda create --name myenv
# RUN conda env create -f environment/kien_env_37.yaml

# Activate the Conda environment
#SHELL ["conda", "run", "-n", "age_venv_37", "/bin/bash", "-c"]


# Set the environment variables for conda virtual environment
# ENV PATH="/opt/conda/envs/age_venv_37/bin:$PATH"

# Test virutal environenment 
# RUN echo "Make sure gunicorn is installed:"
# RUN python -c "import gunicorn"

# The container is listening on port 6080
EXPOSE 6080

# Run the API in docker
# CMD ["gunicorn", "scripts.main:app",  "-w", "1", "--timeout", "180", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0"]
# CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "7090"]
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "7090"]
