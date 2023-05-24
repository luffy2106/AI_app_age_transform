FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04 as base


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
    && apt-get clean

# Install anaconda
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
# RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p $HOME/anaconda3

# Install miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda3
# RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p $HOME/anaconda3


FROM continuumio/miniconda3 as final

# Copy necessary files from the previous stage
COPY --from=base / /

COPY . /app/age_transform_api
WORKDIR /app/age_transform_api

RUN conda env create -f environment/kien_env_37.yaml

# CMD ["/bin/bash"]
# ENV PATH="~/anaconda3/bin:$PATH"

# RUN . ~/anaconda3/bin/activate
# RUN $HOME/anaconda3/bin/conda env create -f environment/kien_env_37.yaml
#RUN echo "conda activate age_venv_37" > ~/.bashrc
#ENV PATH="$HOME/anaconda3/bin:$PATH"
# SHELL ["conda", "run", "-n", "age_venv", "/bin/bash", "-c"]
#RUN conda activate age_venv


# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "age_venv_37", "/bin/bash", "-c"]
RUN echo "Make sure gunicorn is installed:"
RUN python -c "import gunicorn"



EXPOSE 8000

CMD ["conda", "run", "-n", "age_venv_37", "gunicorn", "scripts.main:app",  "-w", "1", "--timeout", "180", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
# CMD ["conda", "run", "-n", "age_venv_37", "/bin/bash", "-c", "gunicorn", "scripts.main:app",  "-w", "1", "--timeout", "180", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]