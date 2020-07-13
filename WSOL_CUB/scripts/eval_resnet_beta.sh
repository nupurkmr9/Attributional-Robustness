gpu=7
name=resnet_beta50_eps2
epoch=200
decay=40
model=resnet50
server=tcp://127.0.0.1:12345
batch=12
wd=1e-4
lr=0.01
data_root="datalist/CUB_200_2011"

CUDA_VISIBLE_DEVICES=${gpu} python3 train_Art.py -a ${model} \
    --resume model_best.pth.tar \
    --data ${data_root} --dataset CUB \
    --train-list datalist/CUB/train.txt \
    --test-list datalist/CUB/test.txt \
    --data-list datalist/CUB/ \
    --task wsol \
    --batch-size ${batch} --name ${name} \
    --beta --evaluate

