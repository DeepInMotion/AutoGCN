AutoGCN
==============================

Implementation of AutoGCN - Towards Generic Human Activity Recognition with Neural Architecture Search.

### Libraries

The code is based on Python >= 3.10 and PyTorch (1.13.0). Run the following command to install all the required packages 
from setup.py:
```
pip install .
```

### Dataset 

The NAS procedure and the experiments are done on the **NTU RGB+D 60 & 120** datasets, which can be downloaded 
[here](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

Furthermore, the Kinetics dataset is used, which can be downloaded 
[here](https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ).

#### Preprocessing

To preprocess, the user is referred to the instructions at:
```
./src/data/preprocess/README.md
```

## Run

To run a model search, check the config file and change the following parameters:

```
work_dir    -> path where you want to store the output of the run
dataset     -> Choose the dataset and change the folders to the preprocessed npy-files
```

The modes can either be activated or deactivated by setting the flags to True or False - refer to the different 
config run files in the ``logs`` folder for further information.

Afterward, execute:
```
python main.py -config /path/to/config.yaml
```

## Results

The results reported in our study are stored in the `./logs` folder.

| Config | Iterations | Top-1 X-View | Top-1 X-Sub |
|--------|------------|--------------|-------------|
| 1007   | 10         | 95.3         | 85.9        |
| 1003   | 20         | 95.1         | 88.3        |
| 1004   | 30         | 95.5         | 86.4        |


There are also predefined configs stored in there, which can be used :).

## Citation and Contact

If you have any questions, feel free to send an email to `felix.e.f.tempel@ntnu.no` or open an issue on GitHub.

Please cite our paper if you use this code in your research. :)
```
cite
```
