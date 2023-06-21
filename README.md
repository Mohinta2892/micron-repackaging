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

WIP:
- [X] Training the UNET to detect the microtubules
- [ ] Replace MongoDB with PostgreSQL 
- [X] Prediction
- [X] Graph generation
- [ ] Solving the ILP

Changes made to ensure that the network can be trained:
- [ ] Changed Lsds import statement in train_pipeline
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
