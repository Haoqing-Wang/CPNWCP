import os
import torch
from torch.utils.data import DataLoader
from data.load_unlabel import UnlabelledDataset

from methods.backbone import model_dict
from methods.simclr import SimCLR
from methods.bartwins import BarTwins
from methods.byol import BYOL
from methods.pn import PN
from methods.cpn import CPN
from methods.cpn_cr import CPN_CR
from methods.cpn_ls import CPN_LS
from methods.cpn_cp import CPN_CP
from methods.cpn_js import CPN_JS
from methods.cpn_wcp import CPN_WCP

from options import parse_args
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


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=60, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train(train_loader, train_model, params):
    log_dir = "output/log/" + params.name + '/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    optimizer = torch.optim.Adam(train_model.parameters(), params.lr)
    scheduler = cosine_scheduler(5e-3, 1e-6, params.epoch, len(train_loader))

    # start
    train_model.train()
    log_file = open(log_dir + 'Train_Log.txt', "w")
    batch_number = len(train_loader)
    print_freq = batch_number // 10 + 1
    for epoch in range(params.epoch):
        avg_loss = 0.
        for i, x in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = scheduler[len(train_loader)*epoch+i]
            optimizer.zero_grad()
            loss = train_model.forward(x.cuda())
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if (i + 1) % print_freq == 0 or (i + 1) == batch_number:
                print('Epoch {:d} | Iterations {:d}/{:d} | Loss {:f}'.format(epoch, i+1, batch_number, avg_loss/float(i+1)))

        print("Epoch %d: %.2f"%(epoch, avg_loss/batch_number))
        print("Epoch %d: %.2f"%(epoch, avg_loss/batch_number), file=log_file)
        if (epoch+1) % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch+1))
            torch.save({'epoch': epoch, 'state': train_model.feature.state_dict()}, outfile)


# --- main function ---
if __name__=='__main__':
    # parser argument
    params = parse_args()
    print('--- Unsupervised Training ---')
    print(params)

    # output and tensorboard dir
    params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- Prepare Dataloader ---')
    train_file = os.path.join(params.data_dir, params.dataset)
    train_dataset = UnlabelledDataset(params.dataset, train_file, aug_num=params.aug_num, backbone=params.backbone)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=12)

    # model
    print('\n--- Build Model ---')
    if params.method == 'BarTwins':
        train_model = BarTwins(model_dict[params.backbone]).cuda()
    elif params.method == 'SimCLR':
        train_model = SimCLR(model_dict[params.backbone]).cuda()
    elif params.method == 'BYOL':
        train_model = BYOL(model_dict[params.backbone]).cuda()
    elif params.method == 'pn':
        train_model = PN(model_dict[params.backbone], params.aug_num).cuda()
    elif params.method == 'cpn':
        train_model = CPN(model_dict[params.backbone], params.aug_num).cuda()
    elif params.method == 'cpn_cr':
        train_model = CPN_CR(model_dict[params.backbone], params.aug_num).cuda()
    elif params.method == 'cpn_ls':
        train_model = CPN_LS(model_dict[params.backbone], params.aug_num, params.alpha).cuda()
    elif params.method == 'cpn_cp':
        train_model = CPN_CP(model_dict[params.backbone], params.aug_num).cuda()
    elif params.method == 'cpn_js':
        train_model = CPN_JS(model_dict[params.backbone], params.aug_num).cuda()
    elif params.method == 'cpn_wcp':
        train_model = CPN_WCP(model_dict[params.backbone], params.aug_num, params.gamma).cuda()
    else:
        print("Please specify the method!")
        assert False

    # training
    print('\n--- Start the training ---')
    train(train_loader, train_model, params)