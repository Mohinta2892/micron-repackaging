##Update 11/14 Aug 2023
Steps:
1. Updating gunpowder==1.2.2 to gunpowder==1.3.0
   Requires augment-nd>=0.1.3 because `deform_augment` has dependencies for 3D elastic deform on this.
   Requires zarr==2.16.0 and numcodecs==0.11.0

2. Install funlib.persistence with `pip install funlib.persistence==0.1.0` since some of the dependencies such as `open_ds, create_ds` have moved from daisy to persistence.

3. Update daisy with `pip install daisy --upgrade` to `daisy==1.0.0` from `daisy==0.3.0`
   For daisy to work and still use DaisyRequestBlocks gunpowder needs to be updated.

At this stage, you should have daisy talking to gunpowder. However, tensorflow will prediction will probably be stuck, though the graph will load on a cuda gpu. Ensure in `worker_config.ini` to have `num_cpus=1`, else it starts looking for as many gpus as cpus. at certain times.
We also use a `write_candidates` files that was reformated during the first reproduction of micron. File copied from `cremi-dev` in the git repo and import statement changed as follows:
`in predict_block.py
from micron.gp import WriteCandidates -->  from write_candidates import *
`

`zarr_write.py` has been customised by adding an else clause to check if the input dataset has dims cdhw or is dhw. Though dataset shape is preconfigured in `predict_blockwise.py` where we pass the `out_dims`. For now have switched off the `out_dims` passing to `prepare_ds` since, the predicted output was not of shape cdhw and zarr.utils `check_array_shape` was failing to assign as below:
`
dataset[1,30,540,540] = data, where data.shape == 30,540,540`

Errors:
`  File "/usr/local/lib/python3.8/dist-packages/gunpowder/nodes/daisy_request_blocks.py", line 123, in __get_chunks
    block.read_roi.get_offset(),
AttributeError: '_GeneratorContextManager' object has no attribute 'read_roi'
`
Solution:
` Inside daisy_request_blocks.py, the block should be acquired using statement:
 with daisy_client.acquire_block() as block

  instead of block = daisy_client.acquire_block()
`
Docker latest with daisy 1.0.0 and gunpowder 1.3.0
`
/media/samia/DATA/mounts/micron-docker/dockers/micron_tf1_2209_py38_14Aug_bumpdaisy.tar
`
##Update - 15 Aug 2023
Daisy now works with `from micron.gp import WriteCandidates` in `predict_block.py`.
Modified **mknet.py** from prediction on `c+_master.h5` such that in input patchsize of the volume is not larger than its dimensions. If it is, daisy gets stuck during prediction.
At this stage, the output zarr file only saves `reduced_maxima`, since this is what is coded to be saved in `predict_block.py`. Though two other datasets `soft_mask` and `derivatives` also get
created in the zarr file. 
MongoDB: A db should be created based on the values of experiment (-e), training number (-t), prediction number (-p) and iteration/checkpoint value (-i). Inside this db we should see a `blocks_predicted`
collection being created after running predict.py 

####Pending for training
Training runs, but the problem of passing `ckpt_save_every` paramater in `train_config.ini` is not being heeded by the code, potentially due to missing it from being read by the configparser.

####Graph building from prediction
1. Updated `build_graph.py` to have funlib.persistence dependencies for MongoDBProvider and `open_ds`. Also, modified `daisy.run_blockwise()` calls by creating `daisy.Task()` as required by v==1.0.0
2. Added to enable import, otherwise fails with pip install
`/usr/local/lib/python3.8/dist-packages/funlib/persistence/graphs/mongodb_graph_provider.py
from __future__ import annotations`
3. Copied `./soft_mask_c+.zarr` manually to this folder structure, since it does not get copied automatically and the path to it in the config is also wrong. **Pending**



