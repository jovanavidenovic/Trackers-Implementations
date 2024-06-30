from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


np.random.seed(0)

__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # added variables for long term tracking
        self.long_term = True
        self.activate_redetection = False
        self.sampling = "normal_grow"
        self.frame_idx = 0
        self.num_frames_not_visible = 0

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        print("Running on device", self.device)

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 16,  # 32
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)

        if self.long_term:
           self.past_ratios = []
    
    @torch.no_grad()
    def update(self, img):
        if self.frame_idx % 50 == 0:
            print("Frame: ", self.frame_idx)
            print("Activate redetection: ", self.activate_redetection)

        # set to evaluation mode
        self.net.eval()

        img_to_draw = img.copy()

        if self.activate_redetection:
            # search for the target on the entire image
            num_samples = 50
            if self.sampling == "random":
                # randomly sample the locations uniformly over the image
                samples_x = np.random.randint(0, img.shape[1], num_samples)
                samples_y = np.random.randint(0, img.shape[0], num_samples)
            elif self.sampling == "normal":
                # sample the locations from a normal distribution
                gauss_std = 10
                # sample around the center
                samples_x = np.random.normal(self.center[1], gauss_std, num_samples)
                samples_y = np.random.normal(self.center[0], gauss_std, num_samples)
            elif self.sampling == "normal_grow":
                # sample the locations from a normal distribution
                # increase the std as the number of frames not visible increases
                gauss_std = 10 + self.num_frames_not_visible * 0.5
                # sample around the center                
                samples_x = np.random.normal(self.center[1], gauss_std, num_samples)
                samples_y = np.random.normal(self.center[0], gauss_std, num_samples)

            samples_pos = np.column_stack((samples_y, samples_x))
            
            x = [ops.crop_and_resize(
                img, pos, self.x_sz,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color) for pos in samples_pos]
        else:
            # search images
            x = [ops.crop_and_resize(
                img, self.center, self.x_sz * f,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color) for f in self.scale_factors]
                    
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        if self.activate_redetection:
            chosen_scale = 1.0
        else:
            chosen_scale = self.scale_factors[scale_id]

        # peak location
        response = responses[scale_id]
        max_resp = max(0, response.max())

        # long-term tracking
        if self.long_term:
            # remove 10x10 neighborhood of the peak response
            # this represents new_response and is used in PSR
            response_height, response_width = response.shape
            response_height = int(response_height)
            response_width = int(response_width)
            response_max_idx = np.argmax(response)
            response_max_idx_x = response_max_idx % response_width
            response_max_idx_y = response_max_idx // response_width
            new_response = response.copy()
            new_response[response_max_idx_y - 5:response_max_idx_y + 5, response_max_idx_x - 5:response_max_idx_x + 5] = 0
            mean_new_resp = np.mean(new_response)
            std_new_resp = np.std(new_response)

            PSR = (max_resp - mean_new_resp) / std_new_resp
            q = max_resp * PSR
            avg_q = np.mean(self.past_ratios)

            ratio = avg_q / q

            ratio_thr = 1.5
            if len(self.past_ratios) > 10 and ratio <= ratio_thr:
                if self.num_frames_not_visible > 0:
                    print("Redetection for # frames: ", self.num_frames_not_visible, self.frame_idx)
                self.activate_redetection = False
                self.num_frames_not_visible = 0
            elif ratio >= ratio_thr: 
                self.activate_redetection = True

            if self.activate_redetection == False:
                self.past_ratios.append(q)
                if len(self.past_ratios) > 50:
                    self.past_ratios.pop(0)
            else: 
                self.num_frames_not_visible += 1

        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        if self.activate_redetection == False:
            # Update parameters only if redetection is not activated

            # locate target center
            disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
            disp_in_instance = disp_in_response * \
                self.cfg.total_stride / self.cfg.response_up
            disp_in_image = disp_in_instance * self.x_sz * \
                chosen_scale / self.cfg.instance_sz
            self.center += disp_in_image

            # update target size
            scale =  (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * chosen_scale   
            self.target_sz *= scale
            self.z_sz *= scale
            self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
        
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        
        sr_box = np.array([
            self.center[1] + 1 - (self.x_sz - 1) / 2,
            self.center[0] + 1 - (self.x_sz - 1) / 2,
            self.x_sz, self.x_sz
        ])
    
        self.frame_idx += 1

        if self.activate_redetection:
            max_resp = 0

        return box, max_resp
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
