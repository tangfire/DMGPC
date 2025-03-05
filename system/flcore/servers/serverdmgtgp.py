import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientdmgv2 import ClientDMGV2
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict
from torch.utils.data import DataLoader


class FedDMGTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(ClientDMGV2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            TGP = Trainable_Global_Prototypes(
                self.num_classes,
                self.server_hidden_dim,
                self.feature_dim,
                self.device
            ).to(self.device)
            save_item(TGP, self.role, 'TGP', self.save_folder_name)
            print(TGP)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.update_TGP()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_coarse_protos = []  # 新增：单独存储粗粒度原型
        self.uploaded_fine_protos = []  # 新增：单独存储细粒度原型

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            client_protos = load_item(client.role, 'protos', client.save_folder_name)

            # 分离粗/细粒度原型
            self.uploaded_coarse_protos.append(client_protos['coarse'])
            self.uploaded_fine_protos.append(client_protos['fine'])

        # 分别计算类间距
        self._compute_class_gap(self.uploaded_coarse_protos, proto_type='coarse')
        self._compute_class_gap(self.uploaded_fine_protos, proto_type='fine')

    def _compute_class_gap(self, protos_list, proto_type):
        # 聚合同类原型
        avg_protos = proto_cluster(protos_list)

        # 计算最小类间距
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)

        print(f'[{proto_type}] Class-wise minimum distance:', self.gap)

    def update_TGP(self):
        TGP = load_item(self.role, 'TGP', self.save_folder_name)
        TGP_opt = torch.optim.SGD(TGP.parameters(), lr=self.server_learning_rate)
        TGP.train()

        def compute_margin(protos_dict, scale=0.5):
            protos = torch.stack(list(protos_dict.values())).to(self.device)
            dist_matrix = torch.cdist(protos, protos)

            # PyTorch对角线填充
            mask = torch.eye(dist_matrix.size(0), dtype=torch.bool, device=dist_matrix.device)
            dist_matrix = dist_matrix.masked_fill_(mask, float('inf'))

            min_dist = dist_matrix.min(dim=1).values
            return min_dist.median() * scale

        # 分别处理粗/细粒度
        for proto_type in ['coarse', 'fine']:
            protos_list = self.uploaded_coarse_protos if proto_type == 'coarse' else self.uploaded_fine_protos

            # 构建（原型，标签）数据集
            proto_label_pairs = []
            for client_protos in protos_list:
                for class_id, proto in client_protos.items():
                    proto_label_pairs.append((proto, class_id))

            # 计算动态边缘
            avg_protos = proto_cluster(protos_list)
            margin = compute_margin(avg_protos) * (0.7 if proto_type == 'coarse' else 0.3)
            margin = margin.to(self.device)


            # 训练全局原型
            proto_loader = DataLoader(proto_label_pairs, self.batch_size, shuffle=True)
            for batch_protos, batch_labels in proto_loader:
                batch_protos = batch_protos.to(self.device)
                batch_labels = batch_labels.to(self.device)
                TGP_opt.zero_grad()

                global_protos = TGP(list(range(self.num_classes)), proto_type)

                sim_matrix = F.cosine_similarity(
                    batch_protos.unsqueeze(1),
                    global_protos.unsqueeze(0),
                    dim=-1
                ).to(self.device)

                # 应用动态边缘
                sim_matrix = sim_matrix - F.one_hot(batch_labels, self.num_classes).to(self.device) * margin

                loss = F.cross_entropy(sim_matrix, batch_labels)
                loss.backward()
                TGP_opt.step()

        print(f'Server loss: {loss.item()}')
        save_item(TGP, self.role, 'TGP', self.save_folder_name)

        # 下发全局原型
        TGP.eval()
        global_protos = {
            'coarse': {c: TGP([c], 'coarse').squeeze(0).detach() for c in range(self.num_classes)},
            'fine': {c: TGP([c], 'fine').squeeze(0).detach() for c in range(self.num_classes)}
        }
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)


def proto_cluster(protos_list):
    """输入应为单粒度原型列表，如 [client1_coarse_protos, client2_coarse_protos]"""
    proto_clusters = defaultdict(list)

    # 遍历每个客户端的原型字典（单粒度）
    for client_protos in protos_list:
        for class_id, proto in client_protos.items():
            proto_clusters[class_id].append(proto)

    # 按类别聚合
    aggregated_protos = {}
    for class_id, protos in proto_clusters.items():
        aggregated_protos[class_id] = torch.mean(torch.stack(protos), dim=0)

    return aggregated_protos


class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        # self.embedings = nn.Embedding(num_classes, feature_dim)
        # layers = [nn.Sequential(
        #     nn.Linear(feature_dim, server_hidden_dim),
        #     nn.ReLU()
        # )]
        # self.middle = nn.Sequential(*layers)
        # self.fc = nn.Linear(server_hidden_dim, feature_dim)

        # 粗粒度全局原型生成器
        self.coarse_emb = nn.Embedding(num_classes, feature_dim)
        self.coarse_proj = nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU(),
            nn.Linear(server_hidden_dim, feature_dim)
        )

        # 细粒度全局原型生成器
        self.fine_emb = nn.Embedding(num_classes, feature_dim)
        self.fine_proj = nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU(),
            nn.Linear(server_hidden_dim, feature_dim)
        )

    def forward(self, class_ids, proto_type='coarse'):
        """支持返回指定类型的全局原型"""
        class_ids = torch.tensor(class_ids, device=self.device)
        if proto_type == 'coarse':
            emb = self.coarse_emb(class_ids)
            return self.coarse_proj(emb)
        elif proto_type == 'fine':
            emb = self.fine_emb(class_ids)
            return self.fine_proj(emb)

    # def forward(self, class_id):
    #     class_id = torch.tensor(class_id, device=self.device)
    #
    #     emb = self.embedings(class_id)
    #     mid = self.middle(emb)
    #     out = self.fc(mid)
    #
    #     return out
