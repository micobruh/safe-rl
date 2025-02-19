# 1. Initialize Python Environment

The python environment should not be 3.11 or higher due to pygame compatibility (Used in Safety-Gymnasium package).

We recommend using python 3.8.

Code to initialize environment:
```
conda create -n safe_env python=3.8
conda activate safe_env
```

# 2. Run in Linux

It is recommended to run this module in Linux for fewer troubleshootings.

Code to run in Linux:
```
wsl
cd
```

# 3. Installing Safety Gymnasium package

This can take some time to install.

Code to install:
```
git clone https://github.com/PKU-Alignment/safety-gymnasium.git
cd safety-gymnasium
pip install -e .
```

# 34. Installing This Package

Code to install:
```
git clone https://github.com/micobruh/safety-rl.git
cd safety-rl
pip install -e .
```
