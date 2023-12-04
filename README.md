AutoGCN
==============================

Implementation of AutoGCN - Towards Generic Human Activity Recognition with Neural Architecture Search for Neural
Architecture Search on Graph Neural Networks for Human Activity Recognition.

### Libraries

Code is based on Python >= 3.10 and PyTorch (1.13.0). Run the following command to install all the required packages 
from setup.py:
```
pip install .
```

### Dataset 

The NAS procedure and the experiments are done on the **NTU RGB+D 60 & 120** datasets, which can be downloaded 
[here](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

Furthermore the Kinetics dataset is used which can be downloaded 
[here](https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ)

#### Preprocessing

To preprocess the user is referred to the instructions at:
```
./src/data/preprocess/README.md
```

## Run

To run a model search check the config file and change following parameters:

```
work_dir    -> path where you want to store the output of the run
dataset     -> choose the dataset and change the folders to the preprocessed npy-files
```

The modes can either be activated or deactivated with setting the flags to True or False - refer to the different 
config run files for further information.

Afterwards execute:
```
python main.py -config ...
```

## Results

The results reported in our study are stored in the `./logs` folder.
There are also predefined configs stored in there, which can be used :).

## Citation and Contact

If you have any question, feel free to send a mail to `felix.e.f.tempel@ntnu.no`.

Please cite our paper if you use this code in your research. :)
```
cite
```
