# micron-repackaging
Nils Eckstein's micron repackaged in a docker env to ensure it runs locally

Path to docker image in docker hub:
```
docker pull mohinta2892/microtubule_tracking_eckstein
```

This docker uses tf==1.15.5 and Python==3.8.10. Runs on gpu. 

The environment for Micron was built on top of the nvidia tf1 docker: 
```
 nvcr.io/nvidia/tensorflow:22.09-tf1-py3
```
Remember all packages in this docker were explicitly as root after running (because the build through dockerfile is unresolved):
```
nvidia-docker run -it -v /local/mount-dir/micron-docker:/home nvcr.io/nvidia/tensorflow:22.09-tf1-py3
```

Torch docker path:
```
/media/samia/DATA/mounts/micron-docker/dockers/micron_torch_tf_20Jun.tar
```

## Training (copied from Nil's Github [repo](https://github.com/nilsec/micron))
##### 1. Training a network:

```
cd micron/micron
python prepare_training.py -d <base_dir> -e <experiment_name> -t <id_of_training_run>
```

This will create a directory at 
```
<base_dir>/<experiment_name>/01_train/setup_t<id_of_training_run> 
```
with all the necessary 
files to train a network that can detect microtubules in EM data.

In order to train a network on your data you need to provide ground truth skeletons and the corresponding raw data.
The paths to the data need to be specified in the provided ```train_config.ini```. Ground truth skeletons should be given
as volumetric data where each skeleton is represented by a corresponding id in the ground truth volume. Raw 
data and ground truth should have the same shape, background should be labeled as zero.

Our training data traced on the 3 CREMI test cubes and raw tracings (Knossos skeletons)
is available [here](https://github.com/nilsec/micron_data.git) and 
can be used for microtubule prediction on FAFB. If you want to train on your own data this can be used as an example
of how to format your data for training. 

An example train_config.ini:
```
training_container = ~/micron_data/a+_master.h5, ~/micron_data/b+_master.h5, ~/micron_data/c+_master.h5
raw_dset = raw
gt_dset = tracing
```
Once the appropriate changes have been made to the train config, network training can be started
via: 
```
python train.py <num_iterations>
```
which will train the network for num_iterations (e.g. 300000) iterations on the provided data and
training checkpoints will be saved every 1000 iterations.



##WIP:
- [X] Training the UNET to detect the microtubules
- [X] Prediction with blockwise daisy
- [X] Graph generation
- [X] Solving the ILP with gurobi and pylp
- [ ] Evaluating the solved trajectories

Changes made to ensure that the network can be trained:
- [X] Changed Lsds import statement in train_pipeline
- [ ] Add a solver package for the ILP to the docker, preferably Cplex since Gurobi's WSL licence has only 90days validity and non-shareable clause?
- [ ] Replace MongoDB with PostgreSQL 
- [ ] More in notes (will weed and add here)

Feature updates:
- [X] Added tqdm to track training progress.
- [X] Added a torch script to train micron
- [ ] Added a torch script to predict with scan nodes - not yet working
      
Upcoming:
- [ ] Fallback to custom gunpowder which supports passing a checkpoint storage folder and allows cuda device to be passed as input.
- [ ] Dockerfile to build docker locally
- [ ] Further notes

**Notes**:
Singularity does not work, have not tried to make it work
