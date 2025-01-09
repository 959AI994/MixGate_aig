from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import os
from config import get_parse_args
import mixgate.top_model
import mixgate.top_trainer 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/dg_pair'

if __name__ == '__main__':
    args = get_parse_args()
    # here,we need to build some npz formate including mig,xmg,xag,aig fusion graph

    # circuit_path = 'datasets/pair_graphs.npz'
    circuit_path ='/home/jwt/MixGate/datasets/merged_all.npz'

    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    # dataset = mixgate.AigParser(DATA_DIR, circuit_path)

    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = mixgate.top_model.TopModel(
        args, 
        # dc_ckpt='./ckpt/dc.pth', 
        dg_ckpt_aig='/home/jwt/MixGate/ckpt/model_aig_gpu.pth',
        dg_ckpt_xag='/home/jwt/MixGate/ckpt/model_xag_gpu.pth',
        dg_ckpt_mag='/home/jwt/MixGate/ckpt/model_xmg_gpu.pth',
        dg_ckpt_mig='/home/jwt/MixGate/ckpt/model_mig_gpu.pth'
    )
    
    trainer = mixgate.top_trainer.TopTrainer(args, model, distributed="True")
    trainer.set_training_args(lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    