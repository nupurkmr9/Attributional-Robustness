# Attributional-Robustness
Training methodology for obtaining attributional robustness and its application in the area of weakly supervised object localization

Our code is derived from the code base of [Attention-based Dropout Layer for Weakly Supervised Object Localization](https://github.com/junsukchoe/ADL) and [A Closer Look at Few-shot Classification](https://github.com/wyharveychen/CloserLookFewShot) 

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


### Running code on CUB for WSOL
------------
```
cd WSOL_CUB/datalist/
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz
cd ..
bash scripts/run_resnet_beta.sh
```



