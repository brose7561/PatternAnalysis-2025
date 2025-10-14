
# Rangpur at Home:

ssh s4743787@remote.labs.eait.uq.edu.au

ssh s4743787@rangpur.compute.eait.uq.edu.au

conda env list

conda activate pytorch-env

srun -p a100-test --gres=shard:1 --time=20:00 --pty bash

streamlit run viewer.py


# local testing

testdata dir containing small sample of total training data for testing reasons. 

python train.py \
--image_dir ./testdata/semantic_MRs \
--label_dir ./testdata/semantic_labels_only \
--epochs 5 \
--batch_size 1 \
--num_workers 0 \
--outdir runs_localtest
