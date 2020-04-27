import tensorflow as tf
import torch
import torch.nn.functional as fn
from torch.autograd import Variable
import os
import random
import numpy as np

import shutil
import datetime
import logging
import sys


stats_headings = [['epoch', '{:>14}', '{:>14d}'],
                  ['errRecon', '{:>14}', '{:>14.3f}'],
                  ['errLatent', '{:>14}', '{:>14.3f}'],
                  ['E_T', '{:>14}', '{:>14.3f}'],
                  ['E_F', '{:>14}', '{:>14.3f}'],
                  ['err(I)', '{:>14}', '{:>14.3f}'],
                  ['err(G)', '{:>14}', '{:>14.3f}'],
                  ['err(E)', '{:>14}', '{:>14.3f}'],
                  ['KLD(z)', '{:>14}', '{:>14.3f}'],
                  ['lr(E)', '{:>14}', '{:>14.6f}'],
                  ['lr(G)', '{:>14}', '{:>14.6f}'],
                  ['lr(I)', '{:>14}', '{:>14.6f}'],
                  ['inc_v2', '{:>14}', '{:>14.3f}'],
                  ['fid_v2', '{:>14}', '{:>14.3f}'],
                ]


# tensorflow
def create_lazy_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    return tf.Session(config=config)


# models
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_flag(netG, netE, netI):
    netG.train()
    netI.train()
    netE.train()


def compute_energy(args, disc_score):
    if args.energy_form == 'tanh':
        energy = torch.tanh(-disc_score.squeeze())
    elif args.energy_form == 'sigmoid':
        energy = fn.sigmoid(-disc_score.squeeze())
    elif args.energy_form == 'identity':
        energy = disc_score.squeeze()
    elif args.energy_form == 'softplus':
        energy = fn.softplus(-disc_score.squeeze())
    return energy


def diag_normal_NLL(z, z_mu, z_log_sigma):
    # define the Negative Log Probability of Normal which has diagonal cov
    # input: [batch nz, 1, 1] squeeze it to batch nz
    # return: shape is [batch]
    nll = 0.5 * torch.sum(z_log_sigma.squeeze(), dim=1) + \
          0.5 * torch.sum((torch.mul(z - z_mu, z - z_mu) / (1e-6 + torch.exp(z_log_sigma))).squeeze(), dim=1)
    return nll.squeeze()


def reparametrize(mu, log_sigma, is_train=True):
    if is_train:
        std = torch.exp(log_sigma.mul(0.5))
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def set_seed(seed):
    assert seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_gpu(device):
    torch.cuda.set_device(device)


def set_cudnn():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def setup_logging(name='main', output_dir='.', console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_exp_id():
    return os.path.splitext(os.path.basename(__file__))[0]


def output_paths(output_dir):
    outf_recon = output_dir + '/recon'
    outf_syn = output_dir + '/syn'
    outf_test = output_dir + '/test'
    outf_ckpt = output_dir + '/ckpt'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(outf_recon, exist_ok=True)
    os.makedirs(outf_syn, exist_ok=True)
    os.makedirs(outf_test, exist_ok=True)
    os.makedirs(outf_ckpt, exist_ok=True)
    return outf_recon, outf_syn, outf_test, outf_ckpt


def update_status(errRecon, errLatent, E_T, E_F, errI, errG, errE, errKld, num_batch):
    global stats_values
    stats_values['errRecon'] += errRecon.data.item() / num_batch
    stats_values['errLatent'] += errLatent.data.item() / num_batch
    stats_values['E_T'] += E_T.data.item() / num_batch
    stats_values['E_F'] += E_F.data.item() / num_batch
    stats_values['err(I)'] += errI.data.item() / num_batch
    stats_values['err(G)'] += errG.data.item() / num_batch
    stats_values['err(E)'] += errE.data.item() / num_batch
    stats_values['KLD(z)'] += errKld.data.item() / num_batch
