#!/usr/bin/env bash

gpus=0
checkpoint_root=checkpoints
# For xBD dataset:
data_name=construction_c
dataset=TestInput
loss=ce
n_class=5
lr=0.001
lr_policy=linear

img_size=1024
batch_size=16
max_epochs=150
net_G=base_transformer_pos_s4_dd8_o5
#net_G=newUNetTrans
# net_G=base_resnet18
#net_G=base_transformer_pos_s4_dd8_dedim8
#net_G=base_transformer_pos_s4
split=train  
split_val=val  
# project_name=CC_all_new_data${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}_ce_smoothen
project_name=model_muneeb



python main_cd_test.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --dataset ${dataset} --loss ${loss} --n_class ${n_class}
# python /home/oem/.local/share/QGIS/QGIS3/profiles/default/python/plugins/atr/CD_pipeline_multi/main_cd_test.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --dataset ${dataset} --loss ${loss} --n_class ${n_class}

