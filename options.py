import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CPNWCP')
    # Training
    parser.add_argument('--dataset', default='miniImagenet', help='miniImagenet/tieredImagenet')
    parser.add_argument('--testset', default='miniImagenet', help='miniImagenet/tieredImagenet')
    parser.add_argument('--method', default='cpn', help='BarTwins/SimCLR/pn/cpn/cpn_cr/cpn_ls/cpn_cp/cpn_js/cpn_wcp')
    parser.add_argument('--classifier', default='ProtoNet', help='ProtoNet/R2D2')
    parser.add_argument('--backbone', default='Conv4', help='Conv4/ResNet10/ResNet12/ResNet18')
    parser.add_argument('--alpha', type=float, default=0.1, help='label relaxation factor')
    parser.add_argument('--gamma', type=float, default=8, help='scaling factor')
    parser.add_argument('--batch_size', default=64, type=int,  help='batch_size')
    parser.add_argument('--aug_num', default=2, type=int, help='number of augmentation images')
    parser.add_argument('--lr', type=float, default=0.001, help='learn rate of unsupervised training')
    parser.add_argument('--data_dir', default='Datasets', type=str, help='')
    parser.add_argument('--save_dir', default='output', type=str, help='')
    parser.add_argument('--name', default='test', type=str, help='')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model')
    parser.add_argument('--epoch', default=600, type=int, help='number of epochs')
    # Evaluation
    parser.add_argument('--n_way', default=5, type=int,  help='class number')
    parser.add_argument('--n_shot', default=5, type=int,  help='number of labeled data in each class as support')
    parser.add_argument('--n_query', default=15, type=int, help='number of labeled data in each class as query')
    return parser.parse_args()