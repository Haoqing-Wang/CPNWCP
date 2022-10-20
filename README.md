# Contrastive Prototypical Network with Wasserstein Confidence Penalty
PyTorch implementation of
<br>
[**Contrastive Prototypical Network with Wasserstein Confidence Penalty**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790654.pdf)
<br>
Haoqing Wang, [Zhi-hong Deng](http://www.cis.pku.edu.cn/jzyg/szdw/dzh.htm)

ECCV 2022


## Prerequisites
- Python >= 3.6
- Pytorch >= 1.2.0 and torchvision (https://pytorch.org/)

## Datasets
For miniImageNet and tiredImageNet, download them from

* **miniImageNet**: https://drive.google.com/file/d/1MnYnUUTHX5KdMIyhIFSYnZ31kTzbGA0C/view?usp=sharing

* **tiredImageNet**: https://drive.google.com/file/d/10l0ev0TF9DjpMTLNzWoCI-nVNhDTpb0C/view?usp=sharing

and put them under their respective paths, e.g., `./Datasets/miniImagenet`.

## Training
Set `method` to `BarTwins`,`SimCLR`,`BYOL`,`pn`, `cpn`, `cpn_cr`, `cpn_ls`, `cpn_cp`, `cpn_js` or `cpn_wcp` for Barlow Twins, SimCLR, BYOL, CPN w/o Pairwise Contrast, CPN, CPN with Consistency Regularization, CPN with Label Smoothing, CPN with Confidence Penalty, CPN with Jensen–Shannon Confidence Penalty or CPN with Wasserstein Confidence Penalty respectively.
```
python train.py --dataset miniImagenet --backbone Conv4 --batch_size 64 --aug_num 4 --method cpn --alpha 0.1 --gamma 8 --name Exp_name
```
`alpha` is the label relaxation factor for Label Smoothing, `gamma` is the scaling factor for Wasserstein Confidence Penalty.

## Evaluation
Set `classifier` to `ProtoNet` for prototype-based nearest-neighbor classifier and to `R2D2` for ridge regression classifier.
```
python test.py --testset miniImagenet --backbone Conv4 --name Exp_name --classifier ProtoNet --n_way 5 --n_shot 5
```

## Calibration
Set `classifier` to `ProtoNet` for prototype-based nearest-neighbor classifier and to `R2D2` for ridge regression classifier.
```
python calibration.py --testset miniImagenet --backbone Conv4 --name Exp_name --classifier ProtoNet --n_way 5 --n_shot 5
```
