import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientdmgbase import ClientDMGBase, load_item, save_item
from collections import defaultdict
from flcore.newmodel.dynprojector import DynamicProjector


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, global_prototypes, labels):
        """
        输入:
            features: [B, d_fused] 本地特征
            global_prototypes: dict {class: [d_fused]}
            labels: [B] 真实类别
        """
        loss = 0.0
        for i in range(features.size(0)):
            c = labels[i].item()
            pos_proto = global_prototypes[c]  # 正样本原型
            neg_protos = [p for k, p in global_prototypes.items() if k != c]

            # 正样本距离
            d_pos = torch.norm(features[i] - pos_proto, p=2)

            # 负样本距离
            d_negs = [torch.norm(features[i] - p, p=2) for p in neg_protos]
            d_neg_min = torch.min(torch.stack(d_negs))

            # 对比损失
            loss += torch.relu(d_pos - d_neg_min + self.margin)
        return loss / features.size(0)

class ClientDMG(ClientDMGBase):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.loss_contrastive = ContrastiveLoss(margin=1.0)


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

                if global_protos is not None:
                    # proto_new = copy.deepcopy(rep.detach())
                    output_fused = projector(outputs['coarse'],outputs['fine'])
                    proto_new = copy.deepcopy(output_fused.detach())
                    # print("proto_new = ", proto_new.shape)
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    # loss += self.loss_mse(proto_new, rep) * self.lamda
                    # print("label = ",y)

                    # loss += self.loss_contrastive(proto_new,global_protos ,y)* self.lamda
                    loss += self.loss_contrastive(proto_new,global_protos ,y)



                for i, yy in enumerate(y):
                    y_c = yy.item()
                    # protos[y_c].append(rep[i, :].detach().data)
                    coarse_features[y_c].append(outputs['coarse'][i,:].detach().data)
                    fine_features[y_c].append(outputs['fine'][i,:].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        coarse_protos = agg_func(coarse_features)
        fine_protos = agg_func(fine_features)

        coarse_classes = list(coarse_protos.keys())
        fine_classes = list(fine_protos.keys())

        P_coarse = torch.stack([coarse_protos[c] for c in coarse_classes])
        P_fine = torch.stack([fine_protos[c] for c in fine_classes])

        P_fused = projector(P_coarse,P_fine)

        # 将P_fused转换为字典结构
        fused_protos = {
            c: P_fused[i] for i, c in enumerate(coarse_classes)
        }


        # save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        save_item(fused_protos,self.role,'protos',self.save_folder_name)
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
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    # rep = model.base(x)
                    outputs = model(x)
                    coarse_rep = outputs['coarse']
                    fine_rep = outputs['fine']
                    rep = projector(coarse_rep,fine_rep)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
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