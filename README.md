# MixGate

python -m torch.distributed.launch --nproc_per_node=5 --master_port=29964 train_mask.py --exp_id mcm_0.03 --batch_size 4 --num_epochs 60 --mask_ratio 0.03 --gpus 0,1,2,3,7