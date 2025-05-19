import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
import os
from ..attack import Attack

from typing import Union


class NB_attack(Attack):
    def __init__(self, model, coord_range:Union[np.ndarray, torch.Tensor], eps=0.3, alpha=2/255, iters=40, use_coord:bool=False, use_color:bool=True):
        super(NB_attack, self).__init__("NB_attack", model)
        self.model = model
        self.eps=eps
        self.alpha=alpha
        self.iters=iters
        self.use_coord = use_coord
        self.use_color = use_color
        assert use_coord or use_color, "a least one type of attack should be selected"
        
        self.coord_range = coord_range
        if isinstance(self.coord_range, np.ndarray):
            self.coord_range = torch.from_numpy(self.coord_range).to(self.device)

    def forward(self, images, labels):
        coord = images[:, :3].clone().detach().to(self.device)
        color = images[:, 3:6].clone().detach().to(self.device)
        ori_image = images.clone().detach().to(self.device)

        adv_images = images.clone().detach().to(self.device)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        loss = nn.CrossEntropyLoss(reduction='sum')

        for i in range(self.iters):
            adv_images = adv_images.detach()
            coord = adv_images[:, :3].detach().clone().requires_grad_(self.use_coord).to(self.device)
            color = adv_images[:, 3:6].detach().clone().requires_grad_(self.use_color).to(self.device)

            if self.use_color:
                adv_images[:, 3:6] = color
                
            if self.use_coord:
                adv_images[:, :3] = coord
            outputs,_ = self.model(adv_images)

            self.model.zero_grad()
            cost = (loss(outputs.reshape(-1, outputs.size(2)), labels.view(-1))/outputs.size(1)).to(self.device)
            cost.backward()

            if self.use_color:
                adv_images[:, 3:6] = adv_images[:, 3:6] + self.alpha*color.grad.sign()
                eta = torch.clamp(adv_images[:, 3:6] - ori_image[:, 3:6], min=-self.eps, max=self.eps)
                color = torch.clamp(ori_image[:, 3:6] + eta, min=0, max=1).detach()
                adv_images[:, 3:6] = color

                
            if self.use_coord:
                adv_images[:, :3] = adv_images[:, :3] + self.alpha*coord.grad.sign()
                eta = torch.clamp(adv_images[:, :3] - ori_image[:, :3], min=-self.eps, max=self.eps)
                coord = ori_image[:, :3] + eta
                adv_images[:, :3] = coord
                adv_images[:, 6:] = coord / self.coord_range[:, None]
                
        dis = torch.dist(adv_images[:, :6], ori_image[:, :6], p=2) #/ batch_size
        return adv_images

from typing import Tuple
from utils.loss_utils import *
from utils.bp_utils import *

class segeidos_attack(Attack):
    def __init__(self, model, coord_range:Union[np.ndarray, torch.Tensor], eps=0.3, alpha=2/255, iters=40, use_coord:bool=False, use_color:bool=True):
        super(segeidos_attack, self).__init__("segeidos_attack", model)
        self.model = model
        self.eps=eps
        self.alpha=alpha
        self.iters=iters
        self.use_coord = use_coord
        self.use_color = use_color
        assert use_coord, "a least one type of attack should be selected"
        
        self.l2_weight = 1.0
        self.hd_weight = 1.0
        self.cd_weight = 1.0
        self.curv_weight = 0.0
        
        self.coord_range = coord_range
        if isinstance(self.coord_range, np.ndarray):
            self.coord_range = torch.from_numpy(self.coord_range).to(self.device)
            
        self.prepare_optim_loss()
            
    def prepare_optim_loss(self):
        self.optims = []
        if self.l2_weight != 0.0:
            self.optims.append(
                loss_wrapper(norm_l2_loss, channel_first=True, keep_batch=True)
            )

        if self.hd_weight != 0.0:
            self.optims.append(
                loss_wrapper(hausdorff_loss, channel_first=True, keep_batch=True)
            )

        if self.curv_weight != 0.0:
            self.optims.append(
                loss_wrapper(
                    local_curvature_loss,
                    channel_first=True,
                    keep_batch=True,
                    need_normal=True,
                )
            )

        if self.cd_weight != 0.0:
            self.optims.append(
                loss_wrapper(pseudo_chamfer_loss, channel_first=True, keep_batch=True)
            )

    def get_loss(
        self, points: torch.Tensor, ori_points: torch.Tensor, normal_vec: torch.Tensor
    ) -> list[torch.Tensor]:
        loss = []
        for optim in self.optims:
            if optim.need_normal:
                loss.append(optim(points, ori_points, normal_vec))
            else:
                loss.append(optim(points, ori_points))

        return loss
            
    def get_delta(
        self, points: torch.Tensor, ori_points: torch.Tensor, normal_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        points = points.detach()
        points.requires_grad = True

        deltas = []

        loss_list = self.get_loss(points, ori_points, normal_vec)

        for loss_mask in range(loss_list.__len__()):
            loss = loss_list[loss_mask].sum()

            if points.grad is not None:
                points.grad.zero_()

            if loss_mask != loss_list.__len__() - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            deltas.append(points.grad.detach().clone())
            
        return torch.stack(loss_list).transpose(0, 1), deltas


    def forward(self, images, labels):
        coord = images[:, :3].clone().detach().to(self.device)
        # color = images[:, 3:6].clone().detach().to(self.device)
        ori_image = images.clone().detach().to(self.device) # [B, C, N]
        adv_images = images.clone().detach().to(self.device)
        
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        loss = nn.CrossEntropyLoss(reduction='sum')

        for i in range(self.iters):
            adv_images = adv_images.detach()
            coord = adv_images[:, :3].detach().clone().requires_grad_(self.use_coord).to(self.device)
            # color = adv_images[:, 3:6].detach().clone().requires_grad_(self.use_color).to(self.device)

            # if self.use_color:
            #     adv_images[:, 3:6] = color
                
            if self.use_coord:
                adv_images[:, :3] = coord
            outputs,_ = self.model(adv_images)

            self.model.zero_grad()
            cost = (loss(outputs.reshape(-1, outputs.size(2)), labels.view(-1))/outputs.size(1)).to(self.device)
            cost.backward()

            # Boundary projection injected.
            g = coord.grad.detach()

            g_norm = (g**2).sum((1, 2)).sqrt()
            g_norm.clamp_(min=1e-12)
            g_hat = g / g_norm[:, None, None]
            # import pdb;pdb.set_trace()
            losses, deltas = self.get_delta(
                adv_images[:, :3], ori_image[:, :3], None
            )  # list[B, 3, n]

            alpha = gram_schmidt(g_hat, deltas, need_gradient=True)  # list[B, n, 3]

            alpha_hat = torch.stack(alpha).sum(0)  # [B, n, 3]


            # if self.use_color:
            #     adv_images[:, 3:6] = adv_images[:, 3:6] + self.alpha*color.grad.sign()
            #     eta = torch.clamp(adv_images[:, 3:6] - ori_image[:, 3:6], min=-self.eps, max=self.eps)
            #     color = torch.clamp(ori_image[:, 3:6] + eta, min=0, max=1).detach()
            #     adv_images[:, 3:6] = color

                
            if self.use_coord:
                assert not torch.isnan(alpha_hat).any()
                adv_images[:, :3] = adv_images[:, :3] + alpha_hat * self.alpha
                eta = torch.clamp(adv_images[:, :3] - ori_image[:, :3], min=-self.eps, max=self.eps)
                coord = ori_image[:, :3] + eta
                adv_images[:, :3] = coord
                adv_images[:, 6:] = coord / self.coord_range[:, None]
                
        dis = torch.dist(adv_images[:, :6], ori_image[:, :6], p=2) #/ batch_size
        return adv_images

class NU_attack(Attack):
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super(NU_attack, self).__init__("NU_attack", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels):
        neighour = 10
        color = images[:, 3:6].clone().detach().to(self.device)
        images = images.clone().detach().to(self.device)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        w_color = self.inverse_tanh_space(color).detach()
        w_color.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device) # 1e10
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        prev_cost = torch.full([self.steps], 1e10)
        optimizer = optim.Adam([w_color], lr=self.lr)


        for step in range(self.steps):
            # Get Adversarial Images
            color = self.tanh_space(w_color)
            adv_images = best_adv_images.clone().detach()
            adv_images[:, 3:6] = color
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs,_ = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()
            Smooth_loss = self.smooth(adv_images, images, neighour).sum()
            cost = f_loss

            cost = f_loss + self.c * Smooth_loss + self.c * L2_loss
            # test acc
            pred = outputs.max(dim=2)[1]
            acc = pred.eq(labels.view_as(pred)).sum().item() / 4096

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            best_adv_images = adv_images.clone().detach()
            # Early Stop when loss does not converge.
            if acc < 1/13:
                return best_adv_images
                if (step > 0) & (step % 100 == 0):
                    lr = lr/2
                    optimizer = optim.Adam([w_pertub], lr=lr)
                    # print("lr oacc cost: ", lr, acc, cost)
                if (step > 10)&(step % 10 == 0):
                    if cost.item() > prev_cost[step-10]:
                        # print("======bingo======")
                        images[:, 3:6][:,:,mask] = images[:, 3:6][:,:,mask] + torch.empty_like(images[:, 3:6][:,:,mask]).uniform_(0, 1)
                        images[:, 3:6] = torch.clamp(images[:, 3:6], min=0, max=1)
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)
        # return self.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        outputs = nn.functional.softmax(outputs, dim=2) # important
        outputs = outputs.transpose(1, 2)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
        one_hot_labels = one_hot_labels.transpose(1, 2)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j, _ = torch.max(one_hot_labels * outputs, dim=1)
        return torch.clamp(self._targeted*(j-i), min=-self.kappa)

    def smooth(self, adv_images, images, neighbour):
        adv_pos = adv_images[0,3:6].transpose(1,0) # [4096,3]
        pos = images[0,3:6].transpose(1, 0)
        dist = torch.cdist(adv_pos, pos)  # [4096, 4096]
        sorted_dist, ind_dist = torch.sort(dist, dim=1)
        return sorted_dist[:,:neighbour]
