import dgl
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dgl.utils import expand_as_pair
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

from traffnet.utils.traffNet_utils import inter_path_embedding, od2pathNum


class PathInputEmbedding(nn.Module):
    def __init__(self, a, b):
        super(PathInputEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=a + 1,
                                      embedding_dim=b).to(torch.device('cuda:0'))

    def forward(self, batch_graph):
        edgeIdOfPath = batch_graph.nodes['path'].data['segmentId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        pathEdgeEmbedding = self.embedding(edgeIdOfPath).to(torch.device('cuda:0'))
        # 在inner-path双向gru推演时候考虑path由edge属性构成特征的影响
        # 1. get inner-path embedding
        pathedgeFeat = batch_graph.nodes['path'].data['pathSegmentFeat']
        pathInputEmbedding = torch.cat([pathEdgeEmbedding, pathedgeFeat], dim=2)

        batch_graph.nodes['path'].data['embedding'] = pathInputEmbedding
        return batch_graph


class InnerPathModel(nn.Module):
    def __init__(self,
                 seq_dim,
                 hidden_size,
                 num_layers,
                 edge_num,
                 seq_max_len):
        super(InnerPathModel, self).__init__()
        self.in_size = seq_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num

        self.gru = nn.GRU(input_size=seq_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bias=True,
                          bidirectional=True).to("cuda:0")

    def forward(self, batch_graph):
        seq = batch_graph.nodes['path'].data['embedding']
        edgeIdOfPath = batch_graph.nodes['path'].data['segmentId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(edgeIdOfPath, dim=1).float()
        package = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False).to("cuda:0")
        results, _ = self.gru(package)
        outputs, lens = pad_packed_sequence(results,
                                            batch_first=True, total_length=seq.shape[1])
        batch_graph.nodes['path'].data['embedding'] = outputs
        return batch_graph


class RouteSelectModel(nn.Module):
    '''
        路径选择模块
    '''

    def __init__(self, in_feats, out_feats, num_heads, seq_max_len):
        super(RouteSelectModel, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.seq_max_len = seq_max_len

        self.gatconv = dglnn.GATConv(in_feats=in_feats,
                                     out_feats=out_feats,
                                     num_heads=num_heads,
                                     allow_zero_in_degree=True).to("cuda:0")

        self.linear = nn.Sequential(nn.Linear(out_feats * num_heads * seq_max_len, 180),
                                    nn.ReLU(),
                                    nn.Linear(180, 150),
                                    nn.ReLU(),
                                    nn.Linear(150, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 80),
                                    nn.ReLU(),
                                    nn.Linear(80, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 1)).to("cuda:0")

        self.linear_trans = nn.Linear(in_features=out_feats * num_heads * seq_max_len,
                                      out_features=1,
                                      bias=False).to("cuda:0")

    def forward(self, batch_graph):
        g_sub = dgl.metapath_reachable_graph(batch_graph, ['select-', 'select+'])
        routeLearningEmbed = g_sub.ndata['embedding'].to(torch.float32)

        # 1.获取路径选择表示
        g_sub = dgl.add_self_loop(g_sub).to("cuda:0")
        _gatEmb = self.gatconv(g_sub, routeLearningEmbed)
        gatEmb = _gatEmb.view(-1, self.out_feats * self.num_heads * self.seq_max_len)
        gatEmb_nor = self.linear(gatEmb)
        wx = self.linear_trans(gatEmb)
        batch_graph.nodes['path'].data['gatEmb'] = gatEmb_nor + wx
        return batch_graph


class PathConcatenation(nn.Module):
    def __init__(self, seq_dim1,seq_dim2,
                 hidden_size,
                 num_layers,
                 edge_num,
                 seq_max_len):
        super(PathConcatenation, self).__init__()
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num
        self.embedding = nn.Embedding(num_embeddings=edge_num + 1,
                                      embedding_dim=seq_dim1).to("cuda:0")
        self.gru = nn.GRU(input_size=seq_dim2,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bias=True,
                          bidirectional=True).to("cuda:0")

    def forward(self, batch_graph):
        edgeIdOfPath = batch_graph.nodes['path'].data['segmentId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(edgeIdOfPath, dim=1).float()

        pathEdgeEmbedding = self.embedding(edgeIdOfPath)
        pathEdgeFeat = batch_graph.nodes['path'].data['pathSegmentFeat']
        pathInputEmbedding = torch.cat([pathEdgeEmbedding, pathEdgeFeat], dim=2)

        g_select = batch_graph.edge_type_subgraph(['select-', 'select+'])
        predFlow = od2pathNum(g_select, False)
        path_flow = predFlow.view(edgeIdOfPath.shape[0], 1, 1).expand(edgeIdOfPath.shape[0], self.seq_max_len, 1)
        pathInputEmbedding[:, :, -1] = path_flow.squeeze()

        package = pack_padded_sequence(pathInputEmbedding, lengths.cpu(), batch_first=True, enforce_sorted=False).to(
            "cuda:0")
        results, _ = self.gru(package)
        outputs, lens = pad_packed_sequence(results,
                                            batch_first=True, total_length=pathInputEmbedding.shape[1])
        batch_graph.nodes['path'].data['embedding'] = outputs
        return batch_graph


class TemporalGRU(nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 num_layers,
                 window_width,
                 batch_size,
                 edge_num,
                 horizon,
                 ):
        super(TemporalGRU, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_width = window_width
        self.batch_size = batch_size
        self.edge_num = edge_num
        self.horizon = horizon

        self.gru = nn.GRU(input_size=in_size * self.edge_num,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True).to(torch.device('cuda:0'))

        self.pred = nn.Sequential(
            nn.Linear(hidden_size, 4600),
            nn.ReLU(),
            nn.Linear(4600, 4000),
            nn.ReLU(),
            nn.Linear(4000, 3600),
            nn.ReLU(),
            nn.Linear(3600, 3000),
            nn.ReLU(),
            nn.Linear(3000, 2400),
            nn.ReLU(),
            nn.Linear(2400, 1800),
            nn.ReLU(),
            nn.Linear(1800, 1300),
            nn.ReLU(),
            nn.Linear(1300, 1000),
            nn.ReLU(),
            nn.Linear(1000, 650),
            nn.ReLU(),
            nn.Linear(650, self.edge_num),
        ).to(torch.device('cuda:0'))

        self.linear_trans = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.edge_num,
                                      bias=False).to(torch.device('cuda:0'))

    def forward(self, batch_graph):
        g_select = batch_graph.edge_type_subgraph(['select-', 'select+'])
        batch_graph.nodes['path'].data['predFlow'] = od2pathNum(g_select)
        path2edge = inter_path_embedding(batch_graph)
        seq = path2edge.reshape(-1, self.window_width, self.in_size * self.edge_num)
        pad = torch.zeros(seq.shape[0], self.horizon - 1, seq.shape[-1]).to(torch.device('cuda:0'))
        pad_seq = torch.hstack((seq, pad))
        all_out, _ = self.gru(pad_seq)
        out = all_out[:, all_out.shape[1] - self.horizon:all_out.shape[1], :].to(
            torch.device('cuda:0'))  # 因为batch_first,要的是最后一个seq
        pred_out = self.pred(out)
        pred_out = pred_out.view(-1, self.horizon, self.edge_num)
        wx = self.linear_trans(out)
        res_out = F.relu(pred_out + wx)
        return res_out
