import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import copy
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.n_users = n_users

    def forward(self, entity_emb,
                edge_index, edge_type,
                weight):

        n_entities = entity_emb.shape[0]

        """aggregate"""
        head, tail = edge_index
        # edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        edge_relation_emb = weight[edge_type]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel] method can be modified here
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        return entity_agg
    
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_relations,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1,agg_type='mixed'):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        self.temperature = 0.2
        self.agg_type = agg_type

        initializer = nn.init.xavier_uniform_
        # weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        weight = initializer(torch.empty(n_relations, channel))
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel] --> [n_relations, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))
            self.W1.append(nn.Linear(channel, channel))
            self.W2.append(nn.Linear(2*channel, channel))
            

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, entity_emb, edge_index, edge_type,
                mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            # interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        # user_res_emb = user_emb  # [n_users, channel]
        # cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb = self.convs[i](entity_res_emb,
                                                 edge_index, edge_type,
                                                 self.weight)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb_1 = torch.add(entity_res_emb, entity_emb)
            entity_res_emb_1 = nn.LeakyReLU()(self.W1[i](entity_res_emb_1))
            entity_res_emb_2 = torch.cat((entity_res_emb,entity_emb), dim=1)
            entity_res_emb_2 = nn.LeakyReLU()(self.W2[i](entity_res_emb_2))
            if self.agg_type == 'add':
                entity_res_emb = entity_res_emb_1
            elif self.agg_type == 'concat':
                entity_res_emb = entity_res_emb_2
            elif self.agg_type == 'mixed':
                entity_res_emb = entity_res_emb_1 + entity_res_emb_2
            else:
                raise NotImplementedError
        relation_res_emb = self.weight
        return entity_res_emb, relation_res_emb

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, field_dims):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        
        # self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.agg_type = args_config.agg
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        # self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        
        self.user_vec = nn.Parameter(self.user_vec)
        self.gcn = self._init_model()
        

## to be modified
        self.mlp_dims = args_config.mlp_dims
        self.dropout = args_config.dfm_dropout
        self.dfm_emb_dim = args_config.dfm_emb_dim
        self.linear = FeaturesLinear(field_dims[:3])
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_dim = (len(field_dims[:3])) * self.dfm_emb_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, self.dropout)
        self.sensi_mlp_dims = args_config.sensi_mlp_dims
        self.attn_output_dim = (3) * self.dfm_emb_dim
        self.attenlayer = MultiLayerPerceptron(self.attn_output_dim, self.sensi_mlp_dims, self.dropout)
        

## to be modified
        self.linear_feat = FeaturesLinear(field_dims)
        self.fm_feat = FactorizationMachine(reduce_sum=True)
        self.embedding_feat = FeaturesEmbedding(field_dims[3:], self.dfm_emb_dim)
        self.embed_output_dim_feat = len(field_dims) * self.dfm_emb_dim
        self.mlp_feat = MultiLayerPerceptron(self.embed_output_dim_feat, self.mlp_dims, self.dropout)



    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.user_vec = initializer(torch.empty(self.n_users,1))
        # # [n_users, n_entities]
        # self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         agg_type=self.agg_type)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, x, test_flag):
        # user = batch['users']
        # item = batch['items']
        

        # user_emb = self.all_embed[:self.n_users, :]
        # item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, relation_gcn_emb = self.gcn(self.all_embed,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        uit_emb = torch.cat((relation_gcn_emb, entity_gcn_emb), dim=0)
        look_up = x.clone()

        look_up = x[:,:3] + torch.tensor([0,self.n_relations, self.n_relations+self.n_users], device=self.device).unsqueeze(0)
        embed_x = F.embedding(look_up, uit_emb)

        b = embed_x.clone()
        c = embed_x.clone()

        # b = embed_x.clone().detach()
        # c = embed_x.clone().detach()
        # b = embed_x.clone().detach()
        # c = embed_x.clone().detach()
        
        attn_weight = self.attenlayer(b.view(-1,self.attn_output_dim))
        # attn_weight = torch.sum(b[:,1,:] * b[:,2,:],dim=1).view(-1,1)
        embed_x[:,1,:] = torch.sigmoid(torch.mul(b[:,1,:], b[:, 0, :])).mul(b[:, 0, :]) + b[:,1,:]
        embed_x[:,2,:] = torch.sigmoid(torch.mul(c[:,2,:], c[:, 0, :])).mul(c[:, 0, :]) + c[:,2,:]
        
        
        out = self.linear(x[:, :3]) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        y_text = torch.sigmoid(out)

## need to be modified 

        embed_x_feat = self.embedding_feat(x[:, 3:])
        embed_x_feat = torch.cat((c, embed_x_feat), dim=1)
        out_feat = self.linear_feat(x) + self.fm_feat(embed_x_feat) + self.mlp_feat(embed_x_feat.view(-1,self.embed_output_dim_feat))
        y_feat = torch.sigmoid(out_feat)
        # user_weight = F.embedding(x[:,1].view(-1,1), self.user_vec).view(-1, 1)
        # user_weight_limited = torch.sigmoid(user_weight)

        attn_weight_limited = torch.sigmoid(attn_weight)
        y_pred = attn_weight_limited * y_text + (1 - attn_weight_limited) * y_feat 
        # y_pred = user_weight_limited * y_text + (1 - user_weight_limited) * y_feat
        # y_pred = y_feat*0.5+y_text*0.5
        # y_pred = y_text
        y_pred = y_pred.squeeze(1)
        
        # return self.create_log_loss(u_e, pos_e, neg_e, cor)
        return y_pred
    
    def generate(self):

        entity_gcn_emb, relation_gcn_emb = self.gcn(self.all_embed,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        
        return relation_gcn_emb, entity_gcn_emb
    # def generate(self):
    #     user_emb = self.all_embed[:self.n_users, :]
    #     item_emb = self.all_embed[self.n_users:, :]
    #     return self.gcn(user_emb,
    #                     item_emb,
    #                     self.latent_emb,
    #                     self.edge_index,
    #                     self.edge_type,
    #                     self.interact_mat,
    #                     mess_dropout=False, node_dropout=False)[:-1]

    # def rating(self, u_g_embeddings, i_g_embeddings):
    #     return torch.sigmoid(torch.matmul(u_g_embeddings, i_g_embeddings.t()))

    # def create_bpr_loss(self, users, pos_items, neg_items, cor):
    #     batch_size = users.shape[0]
    #     pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
    #     neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

    #     mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

    #     # cul regularizer
    #     regularizer = (torch.norm(users) ** 2
    #                    + torch.norm(pos_items) ** 2
    #                    + torch.norm(neg_items) ** 2) / 2
    #     emb_loss = self.decay * regularizer / batch_size
    #     cor_loss = self.sim_decay * cor

    #     return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor