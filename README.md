# RL_tutorials

## Install
### Download code
1. windows
Download ZIP, and extract

2. Linux
```bash
cd <workspace>
git clone https://github.com/0-keun/RL_tutorials.git
```

### virtual environment
creating an environment with commands

```bash
conda create -n env_RL python=3.8
conda activate env_RL
```

### requirements
if you are using virtual environment, you install the requirements after activate the environment.
```bash
pip install -r requirements.txt
```
if it isn't work on Linux, you try this
```bash
cd <right directory>
sudo chmod 777 requirements.txt
```

## Run
you can run the code on command prompt(windows), terminal(linux)

if you don't want to visualize the simulation, change the code
```python
viz = True -> viz = False
```
