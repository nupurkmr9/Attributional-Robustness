# Attributional-Robustness-Training
Training methodology for attributional robustness and its application in weakly supervised object localization

Code for paper: Attributional Robustness Training using Input-Gradient Spatial Alignment (ECCV 2020)

### Requirements
- Python 3.6
- Pytorch (≥ 1.1)
- Python bindings for OpenCV.
- TensorboardX
- OpenCV-python

### Running code on CIFAR-10
------------

```
cd cifar
python cifar_robust_train.py 
```
    
**Evaluation**

    python eval_model.py 


### Running code on CUB-200 for WSOL
------------
```
cd WSOL_CUB
bash scripts/prepare_dataset.sh
bash scripts/run_resnet_beta.sh
```
**Evaluation**
```
bash scripts/eval_resnet_beta.sh
```

### Pretrained models on CIFAR-10 and CUB-200

CIFAR-10: [https://drive.google.com/file/d/1Xjn3kX_Lh887eIKicWhZFBtgRgKpCg6q/view?usp=sharing](https://drive.google.com/file/d/1Xjn3kX_Lh887eIKicWhZFBtgRgKpCg6q/view?usp=sharing)

CUB-200: [https://drive.google.com/file/d/1LMUDHh6deCQ54mpVNqXQnYXXuIiAruzv/view?usp=sharing](https://drive.google.com/file/d/1LMUDHh6deCQ54mpVNqXQnYXXuIiAruzv/view?usp=sharing)
