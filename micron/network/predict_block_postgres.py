"""
Adding postgres to the script

Notes:
    Postgres install on local machine explicitly. Follow icloud cheatsheet
    Postgres access via python through package psycopg2

    Accessing postgres through a docker
Author: Samia Mohinta
"""
from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
import pymongo
import psycopg2
from micron.gp import WriteCandidates


def predict(
        train_setup_dir,
        iteration,
        in_container,
        in_dataset,
        out_container,
        out_dataset,
        db_host,
        db_name,
        run_instruction,
        **kwargs):
    with open('predict_net.json', 'r') as f:
        net_config = json.load(f)

    # voxels
    input_shape = Coordinate(net_config['input_shape'])
    output_shape = Coordinate(net_config['output_shape'])

    # nm
    voxel_size = Coordinate(net_config['voxel_size'])
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = ArrayKey('RAW')
    soft_mask = ArrayKey('SOFT_MASK')
    reduced_maxima = ArrayKey('REDUCED_MAXIMA')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(soft_mask, output_size)
    chunk_request.add(reduced_maxima, output_size)

    if in_container.endswith(('.n5', '.zarr')):

        pipeline = ZarrSource(
            in_container,
            datasets={
                raw: in_dataset
            },
            array_specs={
                raw: ArraySpec(interpolatable=True),
            }
        )
    elif in_container.endswith(('.h5', '.hdf')):
        pipeline = Hdf5Source(
            in_container,
            datasets={
                raw: in_dataset
            },
            array_specs={
                raw: ArraySpec(interpolatable=True),
            }
        )

    print("IN", in_container)

    pipeline += Pad(raw, None)
    pipeline += Normalize(raw)
    pipeline += IntensityScaleShift(raw, 2, -1)

    pipeline += Predict(os.path.join(train_setup_dir,
                                     'train_net_checkpoint_%d' % iteration),
                        inputs={
                            net_config['raw']: raw
                        },
                        outputs={
                            net_config['soft_mask']: soft_mask,
                            net_config['pred_reduced_maxima']: reduced_maxima
                        },
                        graph='predict_net.meta',
                        max_shared_memory=(2 * 1024 * 1024 * 1024),
                        )
    pipeline += IntensityScaleShift(soft_mask, 255, 0)
    print("OUT", out_container)
    pipeline += ZarrWrite(dataset_names={
        soft_mask: out_dataset
    },
        output_filename=out_container
    )

    pipeline += WriteCandidates(maxima=reduced_maxima,
                                db_host=db_host,
                                db_name=db_name)

    pipeline += PrintProfilingStats(every=10)

    pipeline += DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: 'read_roi',
            soft_mask: 'write_roi',
            reduced_maxima: 'write_roi'
        },
        num_workers=run_instruction['num_cache_workers'],
        block_done_callback=lambda b, s, d: block_done_callback(
            db_host,
            db_name,
            run_instruction,
            b, s, d)
        )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")


def block_done_callback(
        db_host,
        db_name,
        run_instruction,
        block,
        start,
        duration,
        sql=True):

    if sql:
        # assumption: a database and a table has already been created in the local postgres via psql or pgadmin
        # Todo - substitute hard-coding with passed arguments
        print("Recording block-done for %s" % (block,))

        client = psycopg2.connect(database="micron_cremi_testdb", user="postgres", password="samia", host='127.0.0.1',
                                  port="5432")
        cur = client.cursor()
        # assume tablename: blocks_predicted
        cur.execute("SELECT * FROM blocks_predicted")
        result = cur.fetchall()

        insert_query = f"INSERT INTO blocks_predicted (block_id, read_roi, write_roi, start, duration)" \
                       f" VALUES ({block.block_id}," \
                       f"{(block.read_roi.get_begin(), block.read_roi.get_shape())}," \
                       f" {(block.write_roi.get_begin(), block.write_roi.get_shape())}," \
                       f"{start}, {duration})"
        cur.execute(insert_query)
        con.commit()

        cur.close()
        con.close()

        print("Recorded block-done for %s" % (block,))

    else:
        print("why am i here?")

        print("Recording block-done for %s" % (block,))

        client = pymongo.MongoClient(db_host)
        db = client[db_name]
        collection = db['blocks_predicted']

        document = dict(run_instruction)
        document.update({
            'block_id': block.block_id,
            'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
            'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
            'start': start,
            'duration': duration
        })

        collection.insert_one(document)

        print("Recorded block-done for %s" % (block,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    worker_instruction = sys.argv[1]

    with open(worker_instruction, 'r') as f:
        worker_instruction = json.load(f)

predict(**worker_instruction)

