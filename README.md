# Object Tracking using Mamba Model

# File Structure
```
image_and_video_analysis/ 
├── main.py
└── data
    ├── train
    ├── test
    ├── val
├── requirements.txt
└── src
    ├── dataset.py
    ├── __init__.py
    ├── model.py
    ├── __pycache__/
    ├── test.py
    ├── train.py
    ├── utils.py
    └── visualize.py
```

# How to setup the virtual environment 
1) Create a conda environment using the command 
 ```sh
   conda create -n mamba_fetrack python=3.10.13
   ```

2) for activating the virtual environment (venv) run command 
Windows : 
 ```sh
   conda activate mamba_fetrack
   ```

For deactivating the venv, run command : 
 ```sh
   conda deactivate
   ```

3) After activating the virtual environment(venv) run this command,
 ```sh
   pip install -r requirements.txt
   ```

4) After this run the code,
 ```sh
   python main.py
   ```

  
