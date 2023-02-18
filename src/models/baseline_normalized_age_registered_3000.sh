#!/bin/sh
source activate organ_mesh_gnn_model

python train.py --model baseline --organ liver --task age_prediction --enc_feats 32 --hidden_channels 256 --use_registered_data True --use_scaled_data True --device 6
python train.py --model baseline --organ right_kidney --task age_prediction --enc_feats 32 --hidden_channels 256 --use_registered_data True --use_scaled_data True --device 6
python train.py --model baseline --organ left_kidney --task age_prediction --enc_feats 32 --hidden_channels 256 --use_registered_data True --use_scaled_data True --device 6
python train.py --model baseline --organ pancreas --task age_prediction --enc_feats 32 --hidden_channels 256 --use_registered_data True --use_scaled_data True --device 6
python train.py --model baseline --organ spleen --task age_prediction --enc_feats 32 --hidden_channels 256 --use_registered_data True --use_scaled_data True --device 6
