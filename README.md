# 1. Initialize Python Environment
The python environment should not be 3.11 or higher due to pygame compatibility (Used in Safety-Gymnasium package).
We recommend using python 3.8.
Code to initialize environment:
```
conda create -n safe_env python=3.8
conda activate safe_env
```

# 2. Installing Safety Gymnasium package
It is important that the python version should not be 3.11 or higher because of pygame compatibility.
We recommend using python 3.8.
Note that pip install the whole package directly is not recommended because the newest version is not in PyPI yet.
Code to install:
```
git clone https://github.com/PKU-Alignment/safety-gymnasium.git
cd safety-gymnasium
pip install -e .
```

# 3. Installing Other Relevant Packages
Run requirements.txt to install all of them.
