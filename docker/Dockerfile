FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04


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

# Create a new Conda environment
# RUN conda create --name myenv
RUN conda env create -f environment/kien_env_37.yaml

# Activate the Conda environment
SHELL ["conda", "run", "-n", "age_venv_37", "/bin/bash", "-c"]

# Set the environment variables for conda virtual environment
ENV PATH="/opt/conda/envs/age_venv_37/bin:$PATH"

# Test virutal environenment 
RUN echo "Make sure gunicorn is installed:"
RUN python -c "import gunicorn"

# Set port of container to connect to the app
EXPOSE 8000

# Run the API in docker
CMD ["gunicorn", "scripts.main:app",  "-w", "1", "--timeout", "180", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]


# Install packages or run commands within the activated environment
# RUN conda install numpy
# RUN python my_script.py




# RUN    apt-get update \
#     && apt-get install -y software-properties-common \
#     && apt-get install wget \
#     && apt-get update \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends build-essential \
#     && apt-get -y install python3-pip \
#     && apt-get install -y libgl1-mesa-dev \
#     && apt-get install -y cmake \
#     && apt-get clean

# Install anaconda
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
# RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p $HOME/anaconda3

# Install miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda3
# RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p $HOME/anaconda3


# FROM continuumio/miniconda3 as final

# # Copy necessary files from the previous stage
# COPY --from=base / /

# COPY . /app/age_transform_api
# WORKDIR /app/age_transform_api

# RUN conda env create -f environment/kien_env_37.yaml

# CMD ["/bin/bash"]
# ENV PATH="~/anaconda3/bin:$PATH"

# RUN . ~/anaconda3/bin/activate
# RUN $HOME/anaconda3/bin/conda env create -f environment/kien_env_37.yaml
#RUN echo "conda activate age_venv_37" > ~/.bashrc
#ENV PATH="$HOME/anaconda3/bin:$PATH"
# SHELL ["conda", "run", "-n", "age_venv", "/bin/bash", "-c"]
#RUN conda activate age_venv


# Make RUN commands use the new environment
# SHELL ["conda", "run", "-n", "age_venv_37", "/bin/bash", "-c"]
# RUN echo "Make sure gunicorn is installed:"
# RUN python -c "import gunicorn"



# EXPOSE 8000

# CMD ["conda", "run", "-n", "age_venv_37", "gunicorn", "scripts.main:app",  "-w", "1", "--timeout", "180", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
# CMD ["conda", "run", "-n", "age_venv_37", "/bin/bash", "-c", "gunicorn", "scripts.main:app",  "-w", "1", "--timeout", "180", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]