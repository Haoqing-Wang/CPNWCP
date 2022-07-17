import os
import torch
from tqdm import tqdm

from methods.backbone import model_dict
from data.load_episode import SetDataManager
from options import parse_args
from methods.protonet import ProtoNet
from methods.r2d2 import R2D2

import random
import numpy as np
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def evaluate(novel_loader, n_way=5, n_support=5):
    log_dir = "output/log/" + params.name + '/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # Model
    if params.classifier == 'R2D2':
        model = R2D2(model_dict[params.backbone], n_way=n_way, n_support=n_support).cuda()
    elif params.classifier == 'ProtoNet':
        model = ProtoNet(model_dict[params.backbone], n_way=n_way, n_support=n_support).cuda()
    else:
        print("Please specify the classifier!")
        assert False

    # Update model
    checkpoint_dir = './%s/checkpoints/%s/%d.tar'%(params.save_dir, params.name, params.epoch)
    state = torch.load(checkpoint_dir)['state']
    print(state.keys())
    model.feature.load_state_dict(state)

    acc_all = []
    model.eval()
    iter_num = len(novel_loader)
    test_log_file = open(log_dir + 'Eval_%dw_%ds.txt'%(params.n_way, params.n_shot), "w")
    for ti, (x, _) in enumerate(tqdm(novel_loader)):
        n_query = x.size(1) - n_support
        model.n_query = n_query
        yq = np.repeat(range(n_way), n_query)
        with torch.no_grad():
            scores = model.set_forward(x.cuda())  # (80, 5)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()  # (80, 1)
            top1_correct = np.sum(topk_ind[:, 0] == yq)
            acc = top1_correct*100./(n_way*n_query)
            acc_all.append(acc)
        print("Task %d: %.1f%%"%(ti, acc), file=test_log_file)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num)))
    print('Test Acc = %4.2f +- %4.2f%%'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num)), file=test_log_file)
    return acc_mean


if __name__=='__main__':
    params = parse_args()
    print(params)

    print('Loading target dataset!')
    iter_num = 10000
    test_file = os.path.join(params.data_dir, params.testset)
    datamgr = SetDataManager(params.testset, n_query=params.n_query, n_way=params.n_way, n_support=params.n_shot, n_eposide=iter_num)
    novel_loader = datamgr.get_data_loader(test_file, 'test')

    evaluate(novel_loader, n_way=params.n_way, n_support=params.n_shot)