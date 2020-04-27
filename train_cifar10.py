import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets, transforms
from .model import _netE, _netG, _netI
from .utils import train_flag, weights_init, compute_energy, stats_headings, \
    reparametrize, diag_normal_NLL, create_lazy_session, \
    get_exp_id, get_output_dir, setup_logging, copy_source, \
    set_gpu, set_cudnn, set_seed, output_paths, update_status


def get_args(exp_id):
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', default=exp_id)

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nez', type=int, default=1, help='size of the output of ebm')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nif', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3, help='number of channels')

    parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')

    parser.add_argument('--niter', type=int, default=1500, help='number of epochs to train for')
    parser.add_argument('--e_lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--g_lr', type=float, default=0.0003, help='learning rate, default=0.0002')
    parser.add_argument('--i_lr', type=float, default=0.0003, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--vfactor', type=float, default=1.0, help='factor for vae component')

    parser.add_argument('--is_grad_clampE', type=bool, default=True, help='whether doing the gradient clamp for E')
    parser.add_argument('--max_normE', type=float, default=100, help='max norm allowed for E')

    parser.add_argument('--is_grad_clampG', type=bool, default=True, help='whether doing the gradient clamp for G')
    parser.add_argument('--max_normG', type=float, default=100, help='max norm allowed for G')

    parser.add_argument('--is_grad_clampI', type=bool, default=True, help='whether doing the gradient clamp for I')
    parser.add_argument('--max_normI', type=float, default=100, help='max norm allowed for I')

    parser.add_argument('--e_decay', type=float, default=0.0000, help='weight decay for EBM')
    parser.add_argument('--i_decay', type=float, default=0.0000, help='weight decay for I')
    parser.add_argument('--g_decay', type=float, default=0.0005, help='weight decay for G')

    parser.add_argument('--e_gamma', type=float, default=0.998, help='lr exp decay for EBM')
    parser.add_argument('--i_gamma', type=float, default=0.998, help='lr exp decay for I')
    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr exp decay for G')

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--netI', default='', help="path to netI (to continue training)")

    return parser.parse_args()


def train(device, args, output_dir, logger):
    # output
    outf_recon, outf_syn, outf_test, outf_ckpt = output_paths(output_dir)

    # data
    dataset = datasets.CIFAR10(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True,
                                             num_workers=int(args.workers))
    dataset_full = np.array([x[0].cpu().numpy() for x in iter(dataset)])
    unnormalize = lambda img: img / 2.0 + 0.5

    # params
    nz, nez, ngf, ndf, nif, nc = int(args.nz), int(args.nez), int(args.ngf), int(args.ndf), int(args.nif), int(args.nc)

    import inception_score_v2_tf as is_v2
    import fid_v2_tf as fid_v2

    netG = _netG(nz, nc, ndf).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))

    netE = _netE(nc, nez, ngf).to(device)
    netE.apply(weights_init)
    if args.netE != '':
        netE.load_state_dict(torch.load(args.netE))

    netI = _netI(nc, nz, nif).to(device)
    netI.apply(weights_init)
    if args.netI != '':
        netI.load_state_dict(torch.load(args.netI))

    input = torch.FloatTensor(args.batchSize, nc, args.imageSize, args.imageSize).to(device)
    noise = torch.FloatTensor(args.batchSize, nz, 1, 1).to(device)
    fixed_noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_().to(device)
    fixed_noiseV = Variable(fixed_noise)
    mse_loss = nn.MSELoss(reduction='sum').to(device)

    optimizerE = optim.Adam(netE.parameters(), lr=args.e_lr, betas=(args.beta1, 0.999), weight_decay=args.e_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.beta1, 0.999), weight_decay=args.g_decay)
    optimizerI = optim.Adam(netI.parameters(), lr=args.i_lr, betas=(args.beta1, 0.999), weight_decay=args.i_decay)

    lrE_schedule = optim.lr_scheduler.ExponentialLR(optimizerE, args.e_gamma)
    lrG_schedule = optim.lr_scheduler.ExponentialLR(optimizerG, args.g_gamma)
    lrI_schedule = optim.lr_scheduler.ExponentialLR(optimizerI, args.i_gamma)

    logger.info(' ')
    logger.info(''.join([h[1] for h in stats_headings]).format(*[h[0] for h in stats_headings]))

    is_v2_score, fid_v2_score = 0., 0.

    num_samples = 50000
    noise_z = torch.FloatTensor(100, nz, 1, 1)
    new_noise = lambda: noise_z.normal_().cuda()

    for epoch in range(args.niter):

        stats_values = {k[0]: 0 for k in stats_headings}
        stats_values['epoch'] = epoch

        lrE_schedule.step()
        lrI_schedule.step()
        lrG_schedule.step()

        num_batch = len(dataloader.dataset) / args.batchSize
        for i, data in enumerate(dataloader, 0):

            train_flag(netG, netE, netI)

            """
            Train EBM 
            """
            netE.zero_grad()
            real_cpu, _ = data
            real_cpu = real_cpu.to(device)
            batch_size = real_cpu.size(0)
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputV = Variable(input)

            disc_score_T = netE(inputV)
            Eng_T = compute_energy(args, disc_score_T)
            E_T = torch.mean(Eng_T)

            noise.resize_(batch_size, nz, 1, 1).normal_()  # or Uniform
            noiseV = Variable(noise)
            samples = netG(noiseV)
            disc_score_F = netE(samples.detach())
            Eng_F = compute_energy(args, disc_score_F)
            E_F = torch.mean(Eng_F)
            errE = E_T - E_F

            errE.backward()
            if args.is_grad_clampE:
                torch.nn.utils.clip_grad_norm_(netE.parameters(), args.max_normE)
            optimizerE.step()

            """
            Train I
            besides the original acd1 which is only build on the generated data, we also consider build on true data
            (1) reconstruct of train data
            (2) kld with prior
            (3) reconstruct of latent codes given generated data
            """
            netI.zero_grad()

            # part 1: reconstruction on train data (May get per-batch loss)
            infer_z_mu_true, infer_z_log_sigma_true = netI(inputV)
            z_input = reparametrize(infer_z_mu_true, infer_z_log_sigma_true)
            inputV_recon = netG(z_input)
            errRecon = mse_loss(inputV_recon, inputV) / batch_size
            errKld = -0.5 * torch.mean(
                1 + infer_z_log_sigma_true - infer_z_mu_true.pow(2) - infer_z_log_sigma_true.exp())

            # part 3: reconstruction on latent z based on the generated data
            infer_z_mu_gen, infer_z_log_sigma_gen = netI(samples.detach())
            errLatent = 0.1 * torch.mean(diag_normal_NLL(noiseV, infer_z_mu_gen, infer_z_log_sigma_gen))

            errI = args.vfactor * (errRecon + errKld) + errLatent
            errI.backward()
            if args.is_grad_clampI:
                torch.nn.utils.clip_grad_norm_(netI.parameters(), args.max_normI)
            optimizerI.step()

            """
            Train G
            besides the original acd1, we add vae criterion which pushes the generator to cover data
            (1) reconstruct the train given re-parameterized z
            (2) MLE of energy and inference: 
                (a) reconstruct the latent space
                (b) Fool the energy discriminator
            """
            netG.zero_grad()
            # part 1: reconstruct the train data
            infer_z_mu_true, infer_z_log_sigma_true = netI(inputV)
            z_input = reparametrize(infer_z_mu_true, infer_z_log_sigma_true)
            inputV_recon = netG(z_input)
            errRecon = mse_loss(inputV_recon, inputV) / batch_size

            # part2: (b): fool discriminator
            disc_score_F = netE(samples)
            Eng_F = compute_energy(args, disc_score_F)
            E_F = torch.mean(Eng_F)

            # part2: (a) : reconstruct the latent space
            infer_z_mu_gen, infer_z_log_sigma_gen = netI(samples)
            errLatent = 0.1 * torch.mean(diag_normal_NLL(noiseV, infer_z_mu_gen, infer_z_log_sigma_gen))

            errG = args.vfactor * errRecon + E_F + errLatent
            errG.backward()
            if args.is_grad_clampG:
                torch.nn.utils.clip_grad_norm_(netG.parameters(), args.max_normG)
            optimizerG.step()

        update_status(errRecon, errLatent, E_T, E_F, errI, errG, errE, errKld, num_batch)

        # images
        if epoch % 10 == 0 or epoch == (args.niter - 1):
            gen_samples = netG(fixed_noiseV)
            vutils.save_image(gen_samples.data, '%s/epoch_%03d_samples.png' % (outf_syn, epoch), normalize=True,
                              nrow=10)

            infer_z_mu_input, _ = netI(inputV)
            recon_input = netG(infer_z_mu_input)
            vutils.save_image(recon_input.data, '%s/epoch_%03d_reconstruct_input.png' % (outf_recon, epoch),
                              normalize=True, nrow=10)

            infer_z_mu_sample, _ = netI(gen_samples)
            recon_sample = netG(infer_z_mu_sample)
            vutils.save_image(recon_sample.data, '%s/epoch_%03d_reconstruct_samples.png' % (outf_syn, epoch),
                              normalize=True, nrow=10)

            # interpolation
            between_input_list = [inputV[0].data.cpu().numpy()[np.newaxis, ...]]
            zfrom = infer_z_mu_input[0].data.cpu()
            zto = infer_z_mu_input[1].data.cpu()
            fromto = zto - zfrom
            for alpha in np.linspace(0, 1, 8):
                between_z = zfrom + alpha * fromto
                recon_between = netG(Variable(between_z.unsqueeze(0).to(device)))
                between_input_list.append(recon_between.data.cpu().numpy())
            between_input_list.append(inputV[1].data.cpu().numpy()[np.newaxis, ...])
            between_canvas_np = np.concatenate(between_input_list, axis=0)
            vutils.save_image(torch.from_numpy(between_canvas_np), '%s/epoch_%03d_interpolate.png' % (outf_syn, epoch),
                              normalize=True, nrow=10, padding=5)

        # metrics
        if epoch > 0 and (epoch % 50 == 0 or epoch == (args.niter - 1)):
            torch.save(netG.state_dict(), outf_ckpt + '/netG_%03d.pth' % epoch)
            torch.save(netI.state_dict(), outf_ckpt + '/netI_%03d.pth' % epoch)
            torch.save(netE.state_dict(), outf_ckpt + '/netE_%03d.pth' % epoch)

            to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))

            gen_samples = torch.cat([netG(new_noise()).detach().cpu() for _ in range(int(num_samples / 100))])
            gen_samples_np = 255 * unnormalize(gen_samples.numpy())
            gen_samples_np = to_nhwc(gen_samples_np)
            gen_samples_list = [gen_samples_np[i, :, :, :] for i in range(num_samples)]

            is_v2_score = is_v2.inception_score(create_lazy_session, gen_samples_list, resize=True, splits=1)[0]
            fid_v2_score = fid_v2.fid_score(create_lazy_session, 255 * to_nhwc(unnormalize(dataset_full)),
                                            gen_samples_np)

        # stats
        stats_values['inc_v2'] = is_v2_score
        stats_values['fid_v2'] = fid_v2_score

        stats_values['lr(G)'] = optimizerG.param_groups[0]['lr']
        stats_values['lr(E)'] = optimizerE.param_groups[0]['lr']
        stats_values['lr(I)'] = optimizerI.param_groups[0]['lr']

        logger.info(''.join([h[2] for h in stats_headings]).format(*[stats_values[k[0]] for k in stats_headings]))

    logger.info('done')


def main():
    # preamble
    exp_id = get_exp_id()
    args = get_args(exp_id)
    output_dir = get_output_dir(exp_id)
    logger = setup_logging(output_dir=output_dir)
    logger.info(args)
    copy_source(__file__, output_dir)

    # device
    device = torch.device(args.device)
    set_gpu(device)
    set_cudnn()
    set_seed(args.seed)

    # go
    train(device, args, output_dir, logger)


if __name__ == '__main__':
    main()
