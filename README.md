# BB-Pose

BB-Pose is a Tensorflow 2 training pipeline for human pose estimation. The aim of this codebase is to set up a foundation on which future projects might be able to build upon. Supported models currently include:

  * **Hourglass Networks** [[1]](https://arxiv.org/abs/1603.06937)
  * **Softgated Skip Connections** [[2]](https://arxiv.org/abs/2002.11098)

Supported datasets currently include:
  * **MPII Dataset** [[3]](http://human-pose.mpi-inf.mpg.de/)
                                                                                
**Note: This repository has only been tested on Ubuntu 18.04 and Debian (Sid).**

## Installation

  1. Clone this repository:
  ```
  git clone https://github.com/BB-Repos/BBpose.git 
  ```
  2. Create a virtual environment (using Pipenv or Conda for example).

  3. Install the project onto your system:
  ```
  pip install -e BBpose
  ```                                                               
  4. Install dependencies:
  ```
  pip install -r BBpose/bbpose/requirements.txt
  ```
  5. Make script executable: 
  ```
  chmod +x BBpose/bbpose/ingestion/scripts/data.sh        
  ```
    
## Params    

This project is setup around a config file which contains numerous adjustable parameters. Any changes to how the project runs must be done here by updating the params. There are 3 main commands to update the params:
  1. The 'reset' command will reset all params back to their default values:
  ```
  python BBpose/bbpose/main.py reset
  ```      
  2. The 'update' command will update all requested params to new values. For example:                 
  ```
  python BBpose/bbpose/main.py update \
    --version 0 \
    --dataset mpii \
    --batch_per_replica 32 \
    --use_records True \
    --num_epochs 200 \
    --num_filters 144 \
  ```
  3. The 'force' command is a special command that will update a set of model-defined hidden params. The command is there for a specific use case (i.e. resetting hidden params after an accidental updating) but in general, users of this repository should never have to use this command.
  ```
  python BBpose/bbpose/main.py force \
    --train_size 22245 \
    --validate_size 2958 \
  ```

**Note #1: The 'reset' command will clear out all user-defined values as well as those of the hidden params. Without these pre-defined params, the model will fail to work properly, if at all. Please use this command carefully.**

**Note #2: CHECK VERSION NUMBER! Users should get into the practice of always checking which version number is being set in order to avoid overwriting old models.**

**Note #3: Do not set the path to 'img_dir' within the data directory. Best to place it in a directory outside of the project.**

There are many params, some of which are interconnected to one another, and some which have limitations. Please see [Glossary](GLOSSARY.md) for a full breakdown of all these params.

That was the hard part. From here on out, the commands to create the datasets, train, and test are simple one liners.

## Datasets

In order to create the datasets, we can make use of the 'ingest' command. This command contains two options:
  1. setup: retrieves required data files and prepares the data for downloading.
  2. generate: downloads the image set and generates Tensorflow datasets.

To setup and start downloading, call:
```
python BBpose/bbpose/main.py ingest --setup --generate
```

**Note: The 'setup' option will clear everything in the data directory. So, if downloading is interrupted, make sure to only use 'generate' to restart downloading.**
```
python BBpose/bbpose/main.py ingest --generate
```

## Training

To start training on a brand new model, call:
```
python BBpose/bbpose/main.py train
```
If training is interupted for any reason, you can easily restart from where you left off:
```
python BBpose/bbpose/main.py train --restore
```

## Testing

To evaluate your trained model:
```
python BBpose/bbpose/main.py test
```

## Contribute

Contributions from the community are welcomed.

## License

BB-Pose is licensed under MIT.

## References

  1. A. Newell, K.Yang, J. Deng, **Stacked Hourglass Networks for Human Pose Estimation**, arXiv:1603.06937, 2016.
  2. A. Bulat, J. Kossaifi, G. Tzimiropoulos, M. Pantic, **Toward Fast and Accurate Human Pose Estimation via Soft-Gated Skip Connections**, arXiv:2002.11098, 2020.



