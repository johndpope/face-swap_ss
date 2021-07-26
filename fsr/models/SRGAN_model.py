import logging
from collections import OrderedDict

import fsr.models.networks as networks
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .base_model import BaseModel

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt, is_train):
        super(SRGANModel, self).__init__(opt, is_train)
        self.rank = 0

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)

        self.print_network()  # print network
        # self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        # ipdb.set_trace()
        self.fake_H = self.netG(self.var_L)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

            pred_g_fake = self.netD(self.fake_H)
            if self.opt.gan_type == 'gan':
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.opt.gan_type == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()
                l_g_gan = self.l_gan_w * (
                        self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        if self.opt.gan_type == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.opt.gan_type == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach().float().cpu()
        out_dict['SR'] = self.fake_H.detach().float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach().float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            print('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                print('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    print('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt.pretrain_model_G
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, True)
        load_path_D = self.opt.pretrain_model_D
        if self.is_train and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, True)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
