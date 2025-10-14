
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



# prediction 

## prediction test

python predict.py \
--image_path ./testdata/semantic_MRs/B006_Week0_LFOV.nii.gz \
--label_path ./testdata/semantic_labels_only/B006_Week0_SEMANTIC.nii.gz \
--checkpoint ./runs_localtest/best.pt



# copy models

## default copy name

scp -r -J s4743787@remote.labs.eait.uq.edu.au \
s4743787@rangpur.compute.eait.uq.edu.au:/home/Student/s4743787/projects/PatternAnalysis-2025/recognition/3D-UNet-Prostate-47437870/runs_3dunet \
/Users/benrose/Desktop/PersonalProjects/PatternAnalysis-2025/recognition/3D-UNet-Prostate-47437870/



