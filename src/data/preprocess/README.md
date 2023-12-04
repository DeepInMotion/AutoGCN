# Data preparation

This folder is for preprocessing the data for the model pipeline.
Currently, NTURGB+D, Kinetics data-sets are supported.


## NTU RGB+D

1. Download the data<br>
Due to the [General Terms](https://rose1.ntu.edu.sg/dataset/actionRecognition/) of the NTU RGB+D database it is not 
allowed to release the annotations. Though, you can download the skeleton data via:

    [Part 1](https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H),
    [Part 2](https://drive.google.com/open?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w)

    For more information refer to the [Paper](https://arxiv.org/pdf/1905.04757.pdf) or to the 
    [Git](https://github.com/shahroudy/NTURGB-D)
    Note: The ignore.txt file has to be provided to make sure that the missing skeletons are excluded.<br>

2. After you have downloaded the data unzip it and put the folder `nturgb+d_skeletons` to `./data/raw/ntu/nturgb+d_skeletons`.

3. Generate the joint dataset first:<br>
4. You can choose between ['ntu60', 'ntu120']
    Before that you have specify the path location in the config file - after that you can run:
    ```bash
   python main_preprocess.py --choice ntu60 --trans true --trans_opt pad sub paralles_s parallel_h view
   ```
   A good guide for the data representation can be found in this blockpost:<br>
    https://lisajamhoury.medium.com/understanding-kinect-v2-joints-and-coordinate-system-4f4b90b9df16

## Kinetics

The [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) skeleton data can be downloaded 
via [GoogleDrive](https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ)
After that you can rebuild the database. For convenience the pytorch dataloader is used.

Simply put your in and out path in kinetics_gen.py and run:
 ```bash
python kinetics_gen.py 
```

