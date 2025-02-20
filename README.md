# Usage Guide on safe-rl

## 1. Initialize Python Environment

The python environment should not be 3.11 or higher due to pygame compatibility (Used in Safety-Gymnasium package).

We recommend using python 3.8.

Code to initialize environment (Replace <safe_env> with the actual environment name):
```
conda create -n <safe_env> python=3.8
conda activate <safe_env>
```

## 2. Run in Linux

It is recommended to run this module in Linux for fewer troubleshootings.

Code to run in Linux:
```
wsl
cd
```

## 3. Installing safety-gymnasium package

This can take some time to install.

Code to install:
```
git clone https://github.com/PKU-Alignment/safety-gymnasium.git
cd safety-gymnasium
pip install -e .
```

To allow global usage of the safety-gymnasium package, include its path in ~/.bashrc file.

### Option 1 (Manual Editing): 

Add the following code at the end of the file (Replace <user_name> with the actual name):
```
export PYTHONPATH=$PYTHONPATH:/home/<user_name>/safety-gymnasium
```

### Option 2 (Command Editing):

Code to do so (Replace <user_name> with the actual name):
```
nano ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/home/<user_name>/safety-gymnasium
```
Save and exit (CTRL+X, Y, ENTER)

Apply the change:
```
source ~/.bashrc
```

## 4. Installing This Package

Code to install:
```
git clone https://github.com/micobruh/safety-rl.git
cd safety-rl
pip install -e .
```
