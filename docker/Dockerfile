FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN    apt-get update \
    && apt-get install -y software-properties-common \
    && apt-get install wget \
    && apt-get update \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get -y install python3-pip \
    && apt-get install -y libgl1-mesa-dev \
    && apt-get install cmake \
    && apt-get clean

# Install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p $HOME/anaconda3
# RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p $HOME/anaconda3

COPY . /app/age_transform_api
WORKDIR /app/age_transform_api

ENV PATH="$HOME/anaconda3/bin:$PATH"

# CMD ["/bin/bash"]
# ENV PATH="~/anaconda3/bin:$PATH"

# RUN . ~/anaconda3/bin/activate
RUN $HOME/anaconda3/bin/conda env create -f environment/kien_env_37.yaml
SHELL ["conda", "run", "-n", "age_venv", "/bin/bash", "-c"]
#RUN conda activate age_venv

EXPOSE 8000

CMD ["gunicorn", "scripts.main:app",  "-w", "1", "--timeout", "180", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]