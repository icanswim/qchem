import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.model import CModel, FFNet

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, global_mean_pool
from torch_geometric.nn import AttentionalAggregation, SAGEConv, GATConv


def pyg_model(model_params):
    """A PyTorch Geometric model launcher"""
    from torch_geometric.nn import models as pygmodels
    
    launcher = getattr(pygmodels, model_params['model_name'])
    model = launcher(**model_params['pyg_params'])
    print('pytorch geometric model {} loaded...'.format(model_params['model_name']))
    return model


class GraphNet(CModel):
    
    def build(self, in_channels=0, hidden=0, out_channels=0):
        
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.lin = Linear(hidden, out_channels)
        print('GNet model loaded...')
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    
class GlobalAttentionNet(CModel):
    
    def build(self, in_channels=0, hidden=0, out_channels=0, depth=0):
        self.layers = []
        self.layers.append(SAGEConv(in_channels, hidden))
        self.layers.append(nn.SELU())
        for d in range(depth):
            self.layers.append(SAGEConv(hidden, hidden)
            self.layers.append(nn.SELU())
        self.attention = AttentionalAggregation(Linear(hidden, 1))
        self.ffu = Cosmo.ff_unit(hidden, hidden, activation=nn.SELU, dropout=.4)
        self.linear = Linear(hidden, out_channels)
                          
    def forward(self, data):
        for l in self.layers:
            X = l(data.x, data.edge.index)
        X = self.attention(X, data.batch)
        X = self.ffu(X)
        X = self.linear(X)
        return F.log_softmax(X, dim=-1)
            
class GlobalAttentionNet(CModel):
    def build(self, dataset, num_layers, hidden):
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.att = AttentionalAggregation(Linear(hidden, 1))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.att(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
        
        
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x