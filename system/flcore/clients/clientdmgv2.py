import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from flcore.clients.clientdmgv2base import ClientDMGV2Base, load_item, save_item
from collections import defaultdict
from flcore.newmodel.dynprojector import DynamicProjector

import random


class MemoryBank:
    def __init__(self, capacity=2048):
        self.buffer = []
        self.capacity = capacity
        self.current_ptr = 0

    def update(self, features, labels):
        # 批量更新特征和标签
        for f, l in zip(features, labels):
            if len(self.buffer) < self.capacity:
                self.buffer.append((f.detach().clone(), l.item()))
            else:
                self.buffer[self.current_ptr] = (f.detach().clone(), l.item())
                self.current_ptr = (self.current_ptr + 1) % self.capacity

    def sample(self, exclude_label, num=32):
        # 排除指定标签并采样
        candidates = [x for x in self.buffer if x[1] != exclude_label]
        return random.sample(candidates, min(num, len(candidates)))


class MultiGranularContrastiveLoss(nn.Module):
    def __init__(self, margins=(0.5, 1.0), temperature=0.5):
        super().__init__()
        self.margin_coarse, self.margin_fine = margins
        self.temperature = temperature

        # 为coarse和fine粒度分别创建记忆库
        self.mem_bank_coarse = MemoryBank(capacity=2048)
        self.mem_bank_fine = MemoryBank(capacity=2048)

    def forward(self, coarse_feat, fine_feat, global_protos, labels):
        # 更新记忆库
        self.mem_bank_coarse.update(coarse_feat, labels)
        self.mem_bank_fine.update(fine_feat, labels)

        # 计算coarse粒度对比损失
        loss_coarse = self._contrastive_loss(
            features=coarse_feat,
            global_protos=global_protos['coarse'],
            labels=labels,
            margin=self.margin_coarse,
            mem_bank=self.mem_bank_coarse,
            temp=self.temperature
        )

        # 计算fine粒度对比损失
        loss_fine = self._contrastive_loss(
            features=fine_feat,
            global_protos=global_protos['fine'],
            labels=labels,
            margin=self.margin_fine,
            mem_bank=self.mem_bank_fine,
            temp=self.temperature
        )

        # 正交损失保持不变
        ortho_loss = self._orthogonality_loss(coarse_feat, fine_feat)
        return loss_coarse + 0.5 * loss_fine + 0.1 * ortho_loss

    def _contrastive_loss(self, features, global_protos, labels, margin, mem_bank, temp):
        # 获取所有类别原型
        classes = sorted(global_protos.keys())
        global_protos_tensor = torch.stack([global_protos[c] for c in classes]).to(features.device)

        # 获取每个样本对应的正原型
        label_indices = torch.tensor([classes.index(l.item()) for l in labels]).to(features.device)
        pos_protos = global_protos_tensor[label_indices]

        # 计算正样本相似度
        sim_pos = torch.cosine_similarity(features, pos_protos, dim=-1) / temp

        # 采样负样本
        neg_samples = []
        for l in labels.unique():
            negs = mem_bank.sample(exclude_label=l.item(), num=32)
            if negs:
                neg_samples.extend([x[0] for x in negs])
        if not neg_samples:
            return torch.tensor(0.0, device=features.device)
        neg_samples = torch.stack(neg_samples).to(features.device)

        # 计算负样本相似度
        sim_neg = torch.cosine_similarity(
            features.unsqueeze(1),
            neg_samples.unsqueeze(0),
            dim=-1
        ) / temp

        # 计算带margin的对比损失
        losses = torch.relu(sim_neg - sim_pos.unsqueeze(1) + margin)
        return losses.mean()

    def _orthogonality_loss(self, feat1, feat2):
        """
        计算两个特征集的正交损失

        Args:
            feat1 (torch.Tensor): 第一组特征
            feat2 (torch.Tensor): 第二组特征

        Returns:
            torch.Tensor: 正交损失
        """
        # 确保输入是张量
        if not isinstance(feat1, torch.Tensor) or not isinstance(feat2, torch.Tensor):
            raise TypeError("Input features must be torch.Tensor")

        # 标准化特征
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # 计算特征间的内积矩阵，并计算Frobenius范数
        return torch.norm(torch.mm(feat1.T, feat2), p='fro')


class ClientDMGV2(ClientDMGV2Base):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.loss_contrastive = MultiGranularContrastiveLoss(margins=(0.8,0.3))# 粗/细粒度边缘参数

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        projector = load_item(self.role, 'projector', self.save_folder_name)
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(projector.parameters()), lr=self.learning_rate)
        # model.to(self.device)

        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        coarse_features = defaultdict(list)
        fine_features = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                # rep = model.base(x)
                # output = model.head(rep)
                outputs = model(x)

                loss = self.loss(outputs['output'], y)

                ortho_loss = outputs['ortho_loss']
                loss+=ortho_loss

                if global_protos is not None:
                    # proto_new = copy.deepcopy(rep.detach())
                    # output_fused = projector(outputs['coarse'], outputs['fine'])
                    # proto_new = copy.deepcopy(output_fused.detach())
                    # proto_new = output_fused
                    # print("proto_new = ", proto_new.shape)
                    # for i, yy in enumerate(y):
                    #     y_c = yy.item()
                    #     if type(global_protos[y_c]) != type([]):
                    #         proto_new[i, :] = global_protos[y_c].data
                    # loss += self.loss_mse(proto_new, rep) * self.lamda
                    # print("label = ",y)

                    # loss += self.loss_contrastive(proto_new, global_protos, y) * self.lamda
                    loss += self.loss_contrastive(outputs['coarse'],outputs['fine'], global_protos, y) * self.lamda

                    # loss += self.loss_contrastive(proto_new,global_protos ,y)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    # protos[y_c].append(rep[i, :].detach().data)
                    coarse_features[y_c].append(outputs['coarse'][i, :].detach().data)
                    fine_features[y_c].append(outputs['fine'][i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        coarse_protos = agg_func(coarse_features)
        fine_protos = agg_func(fine_features)



        # coarse_classes = list(coarse_protos.keys())
        # fine_classes = list(fine_protos.keys())

        # P_coarse = torch.stack([coarse_protos[c] for c in coarse_classes])
        # P_fine = torch.stack([fine_protos[c] for c in fine_classes])
        #
        # P_fused = projector(P_coarse, P_fine)
        #
        # # 将P_fused转换为字典结构
        # fused_protos = {
        #     c: P_fused[i] for i, c in enumerate(coarse_classes)
        # }

        # save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        # save_item(fused_protos, self.role, 'protos', self.save_folder_name)
        # 保存原始粗/细粒度原型（取消投影融合）
        save_item(
            {'coarse': coarse_protos, 'fine': fine_protos},
            self.role,
            'protos',
            self.save_folder_name
        )
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        projector = load_item(self.role, 'projector', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_num = 0

        if global_protos is not None:
            with torch.no_grad():
                # 获取服务器下发的全局融合原型
                fused_global_protos = {}

                for class_id in range(self.num_classes):
                    # 确保原型为 [1, 512]
                    coarse_proto = global_protos['coarse'][class_id].unsqueeze(0)  # 从 [512] -> [1, 512]
                    fine_proto = global_protos['fine'][class_id].unsqueeze(0)

                    # 投影融合
                    fused_proto = projector(coarse_proto, fine_proto)  # [1, 512]
                    fused_global_protos[class_id] = fused_proto.squeeze(0)

                # 计算测试精度
                for x, y in testloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    outputs = model(x)

                    # 本地特征融合
                    fused_local = projector(outputs['coarse'], outputs['fine'])  # [B, 512]

                    # 计算距离矩阵
                    output = torch.zeros(y.shape[0], self.num_classes).to(self.device)
                    for class_id, global_proto in fused_global_protos.items():
                        dist = torch.norm(fused_local - global_proto, dim=1, p=2)
                        output[:, class_id] = dist

                    test_acc += (torch.argmin(output, dim=1) == y).sum().item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/FedDMG/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos