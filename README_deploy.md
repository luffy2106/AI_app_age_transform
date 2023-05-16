# Install conda
Download and Install anaconda in /tmp 
- cd /tmp
- wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
- bash Anaconda3-2023.03-1-Linux-x86_64.sh 
Verify install:
- conda list

# Install virtual environment
Install
- conda env create -f environment/sam_env.yaml
Activate
- conda activate venv


# Install package to run notebook 

If you want to add virutal environment to jupyter
- source activate myenv
- pip install ipykernel
- python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
If you want to see logs of jupyter in vscode : 
- ctlr + shift + P => jupyter : show output

