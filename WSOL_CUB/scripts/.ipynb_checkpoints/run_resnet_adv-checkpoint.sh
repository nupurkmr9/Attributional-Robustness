gpu=6,7
name=resnet_adv_eps2
epoch=2
decay=60
model=resnet50
batch=128
wd=1e-4
lr=0.01
data_root="datalist/CUB_200_2011"

CUDA_VISIBLE_DEVICES=${gpu} nohup python3 -u train_adv.py -a ${model} \
    --pretrained \
    --data ${data_root} --dataset CUB \
    --train-list datalist/CUB/train.txt \
    --test-list datalist/CUB/test.txt \
    --data-list datalist/CUB/ \
    --task wsol \
    --batch-size ${batch} --epochs ${epoch} --LR-decay ${decay} \
    --wd ${wd} --lr ${lr} --nest --name ${name} \
    --beta  > logs/cubs_adv_eps2.txt 

