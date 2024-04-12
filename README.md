# BoXHED Fuse

BoXHED Fuse is a clinical note embedding pipeline designed for BoXHED 2.0, a software package for nonparametrically estimating hazard functions via gradient boosted trees. BoXHED Fuse aims to improve BoXHED 2.0's training capabilities by enriching its dataset with note embeddings.

As a repository, BoXHED Fuse functions independently of BoXHED 2.0. It transforms clinical notes from MIMIC IV electronic health records (EHR) into note embeddings, which are merged with BoXHED 2.0 time-series data. The resulting augmented dataset is directly usable for BoXHED 2.0.

For the BoXHED 2.0 repository, click [here](https://github.com/BoXHED/BoXHED2.0).

For the BoXHED 2.0 paper, please refer to [Pakbin et al. (2023)](#suggested-citations) for details.



## Suggested citations
- TODO add BoXHED Fuse paper

- Pakbin, Wang, Mortazavi, Lee (2023): [BoXHED2.0: Scalable boosting of dynamic survival analysis](https://arxiv.org/abs/2103.12591)

## Prerequisites
- Python (=3.10)
- conda  (we recommend using the free [Anaconda distribution](https://docs.anaconda.com/anaconda/install/))

## How to use BoXHED Fuse -- Overview
1. Obtain data -- Use MIMIC-IV-Extract (TODO LINKME). This is the data we will augment with note embeddings
2. Get embeddings -- BoXHED Fuse will take care of this. The [tutorial](#tutorial) below explains this in detail.
3. Feed into [BoXHED 2.0](https://github.com/BoXHED/BoXHED2.0) -- Use BoXHED Fuse outputs to train and test on BoXHED 2.0



# TUTORIAL

## Obtaining Data
TODO describe permissions for MIMIC IV

## Preprocess Data
TODO link to and explain MIMIC IV EXTRACT


## Setting up BoXHED Fuse

### Environment 
Set up a dedicated virtual environment for BoXHED Fuse. First, create a virtual environment called BoXHED_Fuse:
```
$ conda create -n BoXHED_Fuse python=3.10
```

then activate it
```
$ conda activate BoXHED_Fuse
```

To setup environment, navigate to root "BoXHED_Fuse" directory. Install packages and run setup.py.

```
# Your directory should look something like
# .
# ├── BoXHED_Fuse
# ├── environment.yml
# ├── README.md
# ├── requirements.txt
# └── setup.py

$ python setup.py install 
# This allows module imports

$ pip install -e .
# Any modifications to the BoXHED_Fuse package will automatically be reflected in the build.
```

### Define environmental varaibles

Some scripts contain a --use-wandb argument. If you wish to use wandb to track finetuning progress and store data artifacts, you must obtain a wandb key.

To get your login key, you can head to wandb.ai to create an account. Then go to your settings (https://wandb.ai/settings). In the settings page, you will find your API keys area where you can create a new one or copy an existing one. You will use this key for logging in.

The WANDB_PROJECT_NAME is a name of your choice, which will create a wandb project. Create a unique project name. For example "BoXHED_Fuse".

BHF_ROOT is used to define paths. You can find it by navigating to BoXHED_Fuse/BoXHED_Fuse and entering the command 
```
$ pwd
```

NOTE_DIR is the directroy where your MIMIC IV notes from physionet.org are stored. This may look like some variation of "<your_note_dir>/physionet.org/files/mimic-iv-note/2.2/note/"

In your .bashrc, add the specified environmental variables, shown below. This will automatically be set every time you create a new bash terminal or start a bash script.
```
# add these lines anywhere in your .bashrc
export WANDB_KEY="<your_key>"
export WANDB_PROJECT_NAME="<your_project_name>" 
export BHF_ROOT="<your_root_path>/BoXHED_Fuse/BoXHED_Fuse/"
export NOTE_DIR="<your_note_dir>/

# This only needs to be done once to ensure the environmental variable is set.
    $ source ~/.bashrc

# Verify that the variables are set
    $ echo $WANDB_KEY
    <your_key>
    $ echo $WANDB_PROJECT_NAME
    <your_project_name>
    $ echo $BHF_ROOT
    <your_root_path>/BoXHED_Fuse/BoXHED_Fuse/>   
    $ echo $NOTE_DIR
    <your_note_dir>/physionet.org/files/mimic-iv-note/2.2/note/ 
```
## Running BoXHED Fuse

To add modularity, each python script can be run on its own, by bash script, or within notes_pipeline.py.
I recommend writing an individual bash script for each python script, and then combining them in notes_pipeline.py. 
### OPTION 1: python script
Python scripts are stored in BoXHED_Fuse/src. Navigate there.

```
# to get usage details:
    > python <script_name>.py --h 

# use those arguments to run the file from the terminal. For example,

    > python add_notes.py --test --use-wandb --note-type radiology --noteid-mode recent
```

### OPTION 2: bash script

Bash scripts are stored in BoXHED_Fuse/scripts. Navigate there.

A default bash script is supplied for each python script. 

To change arguments, I recommend creating a new bash script. For example you could create add_notes_1.sh, add_notes_2.sh, etc. 
```
    > bash <script_name>.sh
    
```


### OPTION 3: notes pipeline

This streamlines bash scripts into a single pipeline. Simply supply the bash scripts you intend to use. 

```
# How to use notes pipeline:
$ python notes_pipeline.py --h
usage: notes_pipeline.py [-h] [--bash-files BASH_FILES [BASH_FILES ...]]
                         [--run-scripts RUN_SCRIPTS [RUN_SCRIPTS ...]]

Process bash file names and run scripts.

options:
  -h, --help            show this help message and exit
  --bash-files BASH_FILES [BASH_FILES ...]
                        List of bash file names
  --run-scripts RUN_SCRIPTS [RUN_SCRIPTS ...]
                        List of boolean values indicating whether to run
                        each script
```










<!-- 3. Install the version dependencies by pasting the following lines into your terminal:
```
pip install matplotlib==3.7.1
pip install pillow==9.4.0
pip install numpy==1.24.3
pip install scikit-learn==1.2.2
pip install pytz==2023.3
pip install pandas==1.5.3
pip install cmake==3.26.3
pip install py3nvml==0.2.7
pip install tqdm==4.65.0
pip install threadpoolctl==3.1.0
pip install scipy==1.10.1
pip install joblib==1.2.0
pip install chardet==5.2.0
pip install slicer==0.0.7
pip install numba==0.57.1
pip install cloudpickle==2.2.1
pip install --force-reinstall --upgrade python-dateutil
pip install jupyter
```
If there are any issues with the `pip` installation for any of the packages above, you can use `conda install` to install them instead.

4. **[Mac users only]** Install OpenMP 11.1.0 to enable multithreaded CPU operation:
```
wget https://raw.githubusercontent.com/chenrui333/homebrew-core/0094d1513ce9e2e85e07443b8b5930ad298aad91/Formula/libomp.rb
brew unlink libomp
brew install --build-from-source ./libomp.rb
```
Without OpenMP, BoXHED2.0 will only use a single CPU core, which slows down training and fitting. Also, if OpenMP is not present, setting the variable `nthread` in the tutorial to a value other than 1 may result in a runtime error.

5. Download one of the following pre-built zipped packages for your operating system:
* [BoXHED Linux CPU](https://www.dropbox.com/scl/fi/bi5bkae5ahzedej5gskdl/boxhed_linux_cpu.zip?rlkey=il9zv150xncw5awk9i7hhvzu4&dl=0)
* [BoXHED Linux GPU+CPU](https://www.dropbox.com/scl/fi/f5b51d3njlr61fjpk98w0/boxhed_linux_gpu.zip?rlkey=l41bb5egv9ies5v48mvcs20f2&dl=0)
* [BoXHED Win10 CPU](https://www.dropbox.com/scl/fi/kpz0y8ko7s4aqwdpx5gwu/boxhed_win10_cpu.zip?rlkey=qgy4mkbl78b4vk73tg1m8t32q&dl=0)
* [BoXHED Win10 GPU+CPU](https://www.dropbox.com/scl/fi/wxqfsztoogdsawcev0b6o/boxhed_win10_gpu.zip?rlkey=vc22sgypo9c2oqf2kvkgdhvip&dl=0)
* [BoXHED OSX CPU M1](https://www.dropbox.com/scl/fi/2rztizbhhm7h8rigl2gmb/boxhed_osx_cpu_M1.zip?rlkey=q9232o0pphhd0eoq5ggbiyzhk&dl=0)
  
and place the unzipped contents into the directory returned by the following command: 
```
python -c "import sys; site_packages = next(p for p in sys.path if all([k in p for k in ['boxhed2', 'site-packages']])); print('\n'*2); print(site_packages); print('\n'*2)"
```

For example, the command line above may return the following directory:
```
/home/grads/d/j.doe/anaconda3/envs/boxhed2/lib/python3.8/site-packages/
```

After placing the unzipped contents into this directory, the following folders should exist:
```
/home/grads/d/j.doe/anaconda3/envs/boxhed2/lib/python3.8/site-packages/boxhed/
/home/grads/d/j.doe/anaconda3/envs/boxhed2/lib/python3.8/site-packages/boxhed_kernel/
/home/grads/d/j.doe/anaconda3/envs/boxhed2/lib/python3.8/site-packages/boxhed_prep/
/home/grads/d/j.doe/anaconda3/envs/boxhed2/lib/python3.8/site-packages/boxhed_shap/
```

6. Download the files in this repository and put them in a directory called BoXHED2.0. Then go to the directory:
```
cd BoXHED2.0
```

7. Run *BoXHED2_tutorial.ipynb* for a demonstration of how to fit a BoXHED hazard estimator:
```
jupyter notebook BoXHED2_tutorial.ipynb
```
For Mac users, Apple's security system may complain about the precompiled components of BoXHED2.0. In that case, the instructions on [this page](https://www.easeus.com/computer-instruction/apple-cannot-check-it-for-malicious-software.html) will be helpful. -->
