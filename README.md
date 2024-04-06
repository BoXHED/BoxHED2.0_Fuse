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


## Setting up BoXHED Fuse

2. Set up a dedicated virtual environment for BoXHED Fuse. First, create a virtual environment called BoXHED_Fuse:
```
conda create -n BoXHED_Fuse python=3.10
```

then activate it
```
conda activate BoXHED_Fuse
```

install
```
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
