
# Rangpur at Home:

ssh s4743787@remote.labs.eait.uq.edu.au

ssh s4743787@rangpur.compute.eait.uq.edu.au

conda env list

conda activate pytorch-env

srun -p a100-test --gres=shard:1 --time=20:00 --pty bash

streamlit run viewer.py