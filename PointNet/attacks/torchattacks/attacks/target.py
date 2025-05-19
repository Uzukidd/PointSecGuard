import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from torch.optim import lr_scheduler
from ..attack import Attack

from typing import Union

class tar_NB_attack(Attack):
    def __init__(self, model, coord_range:Union[np.ndarray, torch.Tensor], eps=0.3, alpha=2/255, iters=40,target=None, mask=None, use_coord:bool=False, use_color:bool=True):
        super(tar_NB_attack, self).__init__("tar_NB_attack", model)
        self.model = model
        self.eps=eps
        self.alpha=alpha
        self.iters=iters
        self.target = target
        self.mask = mask

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

        target_labels = torch.full(labels.shape, self.target).to(self.device)

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
            cost = loss(outputs.reshape(-1, outputs.size(2)), target_labels.view(-1)/outputs.size(1)).to(self.device)
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
        dis = torch.dist(adv_images, ori_image, p=2) #/ batch_size
        return adv_images

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




class tar_NU_attack(Attack):
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01, target=None, mask=None):
        super(tar_NU_attack, self).__init__("tar_NU_attack", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.target = target
        self.mask = mask

    def forward(self, images, labels):
        neighour = 5
        color = images[:, 3:6][:,:,self.mask].clone().detach().to(self.device)
        images = images.clone().detach().to(self.device)
        w_color = self.inverse_tanh_space(color).detach()
        w_color.requires_grad = True
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device) # 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        prev_cost = torch.full([self.steps], 1e10)
        optimizer = optim.Adam([w_color], lr=self.lr)

        for step in range(self.steps):
            color = self.tanh_space(w_color)
            adv_images = best_adv_images.clone().detach()
            adv_images[:,3:6][:,:,self.mask] = color
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs,_ = self.model(adv_images)
            if self.target == None:
                f_loss = self.non_f(outputs, labels)
            else:
                f_loss = self.tar_f(outputs, labels)

            Smooth_loss = self.smooth(adv_images, images, neighour).sum()

            cost = f_loss + self.c * Smooth_loss + self.c * L2_loss
            prev_cost[step] = cost
            pred = outputs.max(dim=2)[1]
            acc = pred.eq(labels.view_as(pred)).sum().item() / 4096
            target_labels = torch.full(labels.shape, self.target).to(self.device)

            if self.target == None:
                target_acc = pred[:,self.mask].eq(labels[:,self.mask].view_as(pred[:,self.mask])).sum().item() / self.mask.sum().item()
            else:
                target_labels = torch.full(labels.shape, self.target).to(self.device)
                target_acc = pred[:,self.mask].eq(target_labels[:,self.mask].view_as(pred[:,self.mask])).sum().item() / self.mask.sum().item()
            other_acc = pred[0, ~self.mask].eq(labels[0, ~self.mask].view_as(pred[0, ~self.mask])).sum().item() / (~self.mask).sum().item()

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            best_adv_images = adv_images.clone().detach()


            color_dis_mask = torch.dist(images[:, 3:6][:,:,self.mask], best_adv_images[:, 3:6][:,:,self.mask], p=2)
            color_dis_nmask = torch.dist(images[:, 3:6][:,:, ~self.mask], best_adv_images[:, 3:6][:,:, ~self.mask], p=2)

            if self.target == None:
                if target_acc < 1 / 13: # 1/13
                    return best_adv_images
            else:
                if target_acc > 0.9:
                    return best_adv_images

            if (step > 0) & (step % 50 == 0):
                self.lr = self.lr/2
                optimizer = optim.Adam([w_color], lr=self.lr)

            if (step > 10)&(step % 10 == 0):
                # print(cost.item(), prev_cost[step-10].item())
                if cost.item() >= prev_cost[step-10]:
                    # print("======bingo======")
                    best_adv_images[:, 3:6][:,:,self.mask] = best_adv_images[:, 3:6][:,:,self.mask] + torch.empty_like(best_adv_images[:, 3:6][:,:,self.mask]).uniform_(0, 1)
                    best_adv_images = torch.clamp(best_adv_images, min=0, max=1)
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)


    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)
        # return self.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))



    def non_f(self, outputs, labels):
        outputs = nn.functional.softmax(outputs, dim=2) # important
        outputs = outputs.transpose(1, 2)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
        one_hot_labels = one_hot_labels.transpose(1, 2)
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j, _ = torch.max(one_hot_labels * outputs, dim=1)
        return torch.clamp(self._targeted*(j-i), min=-self.kappa).sum()


    def tar_f(self, outputs, labels):
        outputs = nn.functional.softmax(outputs, dim=2) # important
        outputs = outputs.transpose(1, 2)
        target_labels = torch.full(labels.shape, self.target).to(self.device)
        one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(self.device)

        one_hot_labels = one_hot_labels.transpose(1, 2)
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j, _ = torch.max(one_hot_labels * outputs, dim=1)
        return torch.clamp(self._targeted*(j-i), min=-self.kappa).sum()

    def smooth(self, adv_images, images, neighbour):
        adv_pos = adv_images[0,3:6].transpose(1,0) # [4096,3]
        pos = images[0,3:6].transpose(1, 0)
        dist = torch.cdist(adv_pos, pos)  # [4096, 4096]
        sorted_dist, ind_dist = torch.sort(dist, dim=1)
        return sorted_dist[:,:neighbour]
