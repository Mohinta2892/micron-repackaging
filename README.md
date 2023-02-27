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

WIP:
- [X] Training the UNET to detect the microtubules
- [ ] Prediction
- [ ] Solving the ILP

Changes made to ensure that the network can be trained:
- [ ] Changed Lsds import statement in train_pipeline
- [ ] More in notes (will weed and add here)

Feature updates:
- [ ] Added tqdm to track training progress.

Upcoming:
- [ ] Fallback to custom gunpowder which supports passing a checkpoint storage folder and allows cuda device to be passed as input.
- [ ] Dockerfile to build docker locally
- [ ] Futher notes

**Notes**:
Singularity does not work, have not tried to make it work
