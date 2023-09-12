import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.model import CModel, FFNet

from torch import nn
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
    
    def build(self, in_channels=0, hidden=0, out_channels=0, depth=0, 
                      Conv=GCNConv, pool=global_mean_pool):
        self.layers = []
        self.layers.append(Conv(in_channels, hidden))
        for d in range(depth):
            self.layers.append(Conv(hidden, hidden))
        self.pool = pool
        self.ffn = FFNet({'in_channels':hidden, 'hidden':2*hidden, 'out_channels':out_channels})
                               
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for l in self.layers:
            x = l(x, edge_index)
            x = F.relu(x)
        if self.pool is not None:
            x = self.pool(x, data.batch)
        x = self.ffn(x)
        return x
    
class GlobalAttentionNet(CModel):
    
    def build(self, in_channels=0, hidden=0, out_channels=0, depth=0):
        self.layers = []
        self.layers.append(SAGEConv(in_channels, hidden, normalize=True))
        for d in range(depth):
            self.layers.append(SAGEConv(hidden, hidden, normalize=True))
        self.attention = AttentionalAggregation(CModel.ff_unit(self, hidden, 1, 
                                   activation=None, dropout=None, batch_norm=False))
        self.ffu1 = CModel.ff_unit(self, hidden, hidden, 
                                   activation=nn.ReLU, dropout=.2, batch_norm=True)
        self.ffu2 = CModel.ff_unit(self, hidden, out_channels, 
                                   activation=None, dropout=None, batch_norm=False)
                          
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for l in self.layers:
            x = l(x, edge_index)
            x = F.relu(x)
        x = self.attention(x, data.batch)
        x = self.ffu1(x)
        x = self.ffu2(x)
        return F.log_softmax(x, dim=-1)
