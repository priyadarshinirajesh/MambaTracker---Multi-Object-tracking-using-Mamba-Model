# image_and_video_analysis
Object Tracking using Mamba Model

#File Structure

image_and_video_analysis/
├── weights/
│   └── yolox_s.pth               # YOLOX model weights
├── generate_detection_yolox.py  # Script to generate detections
├── tracker.py                   # Likely for tracking
├── utils.py                     # Helper functions
├── visualize_tracking.py        # For converting outputs to video
├── yolox_exps/                  # Custom YOLOX experiment configs (optional)
├── data/
│   └── test/
│       └── dancetrack0003/
│           └── img1/            # Contains images (frames)


# How to setup the virtual environment 
1) Create a conda environment using the command 
 ```sh
   conda create -n myenv python=3.9
   ```

2) for activating the virtual environment (venv) run command 
Windows : 
 ```sh
   conda activate myenv
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
   python generate_detection_yolox.py
   ```

5) After this run this code,
 ```sh
   python tracker.py
   ```

6) Now to visualize the tracking run this,
 ```sh
   python visualize_tracking.py
   ```