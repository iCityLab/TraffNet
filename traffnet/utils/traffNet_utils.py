import dgl
import dgl.function as fn
import torch
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from torch import nn


def od2pathNum(graph, get_selectPro=False):
    with graph.local_scope():
        aggfn = fn.copy_u('gatEmb', out='he')
        graph.apply_edges(aggfn, etype='select-')

        edata_dict = {('od', 'select+', 'path'): graph.edata['he'][('path', 'select-', 'od')],
                      ('path', 'select-', 'od'): graph.edata['he'][('path', 'select-', 'od')]}
        selectProb = edge_softmax(graph, edata_dict)[('path', 'select-', 'od')]

        graph.edges['select+'].data['selectProb'] = selectProb

        graph.multi_update_all({'select+': (fn.u_mul_e('odNum', 'selectProb', 'm'),
                                            fn.sum('m', 'od2PathNum'))}, 'sum')

        rst = graph.nodes['path'].data['od2PathNum']
        if get_selectPro == True:
            return rst, selectProb
        else:
            return rst


def inter_path_embedding(batch_graph):
    G_predict = dgl.metapath_reachable_graph(batch_graph, ['pass+'])
    orderInfo = batch_graph.edges['pass+'].data['orderInfo']
    edge_feats = batch_graph.nodes['segment'].data['feature'].float()
    pathEmbedding = batch_graph.nodes['path'].data['embedding']
    feat = (pathEmbedding, edge_feats)
    with G_predict.local_scope():
        G_predict.apply_edges(fn.copy_u(u='predFlow', out='_edge_weight'))
        aggregate_fn = MessageFunc(orderInfo=orderInfo)
        feat_src, feat_dst = expand_as_pair(feat, G_predict)

        G_predict.srcdata['h'] = feat_src
        G_predict.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
        rst = G_predict.dstdata['h']
        return rst.squeeze(dim=1)


class MessageFunc(nn.Module):
    '''
        次序聚合中的消息传递函数的构造
    '''

    def __init__(self, orderInfo):
        super(MessageFunc, self).__init__()
        self.orderInfo = orderInfo

    def getMessageFun(self, feat_src, orderInfo):
        unbind_feat_src = torch.unbind(feat_src)
        unbind_orderInfo = torch.unbind(orderInfo)
        messageList = list(
            map(lambda x: torch.index_select(input=x[0], dim=0, index=x[1]),
                tuple(zip(unbind_feat_src, unbind_orderInfo))))
        mailboxInfo = torch.stack(messageList).view(-1, feat_src.shape[2])
        return mailboxInfo

    def forward(self, edges):
        feat_src = edges.src['embedding']  # 根据有链接的边，获得path表示
        mask_node_feat = self.getMessageFun(feat_src=feat_src,
                                            orderInfo=self.orderInfo)
        return {'m': mask_node_feat * edges.data['_edge_weight']}
