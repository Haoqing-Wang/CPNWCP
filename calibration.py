import os
import numpy as np
import torch
from tqdm import tqdm

from methods.backbone import model_dict
from data.load_episode import SetDataManager
from options import parse_args
from methods.protonet import ProtoNet
from methods.r2d2 import R2D2


def evaluate(novel_loader, n_way=5, n_support=5, n_query=16):
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
    model.n_query = n_query

    # Update model
    checkpoint_dir = '%s/checkpoints/%s/%d.tar'%(params.save_dir, params.name, params.epoch)
    state = torch.load(checkpoint_dir)['state']
    print(state.keys())
    model.feature.load_state_dict(state)
    model.eval()

    calib_log_file = open(log_dir + 'Calib_%dw_%ds.txt'%(params.n_way, params.n_shot), "w")
    with torch.no_grad():
        yq = np.repeat(range(n_way), n_query)
        Conf, Acc = [], []
        for i, (x, _) in enumerate(tqdm(novel_loader)):
            scores = model.set_forward(x)
            prob = scores.softmax(-1)  # [80, 5]
            Conf.extend(list(prob.max(1)[0].cpu().numpy()))
            Acc.extend(list(prob.max(1)[1].cpu().numpy()==yq))
        Conf = np.array(Conf)
        Acc = np.array(Acc)

        num, conf, acc = [0, 0], [0., 0.], [0., 0.]
        split = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        for s in split:
            ind = (Conf > s-0.1) & (Conf <= s)
            num.append(ind.sum())
            if num[-1] == 0:
                conf.append(0.)
                acc.append(0.)
            else:
                conf.append(Conf[ind].mean())
                acc.append(Acc[ind].mean())
        num = np.array(num)
        conf = np.array(conf)
        acc = np.array(acc)

        ECE = (num*abs(acc-conf)).sum()/num.sum()
        MCE = np.max(abs(acc-conf))

    print('ECE:%.3f' % ECE, file=calib_log_file)
    print('MCE:%.3f' % MCE, file=calib_log_file)
    print('Num:\n', list(num), file=calib_log_file)
    print('Confidence:\n', list(conf), file=calib_log_file)
    print('Acc:\n', list(acc), file=calib_log_file)


if __name__=='__main__':
    params = parse_args()
    print(params)

    print('Loading target dataset!')
    iter_num = 10000
    test_file = os.path.join(params.data_dir, params.testset)
    datamgr = SetDataManager(params.testset, n_query=params.n_query, n_way=params.n_way, n_support=params.n_shot, n_eposide=iter_num)
    novel_loader = datamgr.get_data_loader(test_file, 'test')

    evaluate(novel_loader, n_way=params.n_way, n_support=params.n_shot, n_query=params.n_query)