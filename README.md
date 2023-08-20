# PointPoseNet

PointPoseNet is a neural network for pose estimation by just one point 

## Installation

### Install the environment

Use the Anaconda.

```bash
conda create -n pointpose python=3.9
conda activate pointpose
```

Use pip to install all libraries and packages with their dependencies
```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py 
usage: main.py [-h] [--dataset_directory DIRECTORY] 
                    [--imagery_dir_name DIR_NAME]                     
                    [--batch_size BATCH_SIZE] 
                    [--image_path IMAGE_PATH]
                    [--single_task TASK_NAME] 
                    [--visualise BOOL] 
``` 

### Usage examples
#### Option 1: Run model on just one dataset

``` bash
python main.py -ds_dir <TASK_PATH> -dir <TASK_NAME> -bs <BATCH_SIZE>
# example
# python main.py -ds_dir /home/user/Downloads/tasks -dir squirrels_head
```

#### Option 2: Run model all datasets

``` bash
python main.py -ds_dir <TASKS_PATH> -dir <TASK_NAME> -bs <BATCH_SIZE>
# example
# python main.py -ds_dir /home/user/Downloads/tasks -dir all
```

#### Option 3: Demo model for single image with visualisation

``` bash
python main.py -im <IMAGE_PATH> -vis 
# example
# python main.py -im /home/user/Pictures/image.jpg
```
