import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.model import CModel, FFNet

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import models as pygmodels
from torch_geometric.nn import GCNConv, Linear, global_mean_pool
from torch_geometric.nn import AttentionalAggregation, SAGEConv, GATConv


class PygModel(nn.Module):
    """
    A PyG model wrapper
    
    model_name = 'ModelName'
    ffnet = True/False
    in_channels = int (ffnet)
    hidden = int (ffnet)
    out_channels = int (ffnet)
    depth = int (num of conv layers)
    pool = 'None'/'global_mean'
    softmax = True/False
    pyg_params = {'in_channels': int,
                  'hidden_channels': int,
                  'num_layers': int,
                  'out_channels': int,
                  'dropout': .int,
                  'norm': None}
    """
    
    def __init__(self, model_params):
        super().__init__()
        Pool = {'global_mean': global_mean_pool, 'None': None}
        launcher = getattr(pygmodels, model_params['model_name'])
        self.model = launcher(**model_params['pyg_params'])
        
        self.softmax = model_params['softmax']
        self.pool = Pool[model_params['pool']]
        self.ffnet = model_params['ffnet']
        
        if self.ffnet:
            self.ffn = FFNet({'in_channels': model_params['in_channels'], 
                              'hidden': model_params['hidden'],
                              'out_channels': model_params['out_channels']})
        
        print('pytorch geometric model {} loaded...'.format(model_params['model_name']))
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        if self.pool is not None:
            x = self.pool(x, data.batch)
        if self.ffnet: 
            x = self.ffn(x)    
        if self.softmax: 
            return F.log_softmax(x, dim=1)
        else:
            return x
        
class GraphNet(CModel):
    """
    builds custom PyG conv nets
    
    
    """

    def build(self, in_channels=0, hidden=0, out_channels=0, depth=0, 
              conv='SAGEConv', pool='global_mean', dropout=.1, 
              softmax=False, activation=F.relu):
        
        Conv = {'GCNConv': GCNConv, 'SAGEConv': SAGEConv, 'GATConv': GATConv}
        Pool = {'global_mean': global_mean_pool, 'None': None}
        
        self.dropout = dropout
        self.activation = activation
        layers = []
        layers.append(Conv[conv](in_channels, hidden))
        for d in range(depth):
            layers.append(Conv[conv](hidden, hidden, normalize=True))
        self.layers = layers
        self.pool = Pool[pool]
        self.ffnet = FFNet({'in_channels':hidden, 'hidden':2*hidden, 'out_channels':out_channels})
        self.softmax = softmax
        print('GraphNet {} loaded...'.format(conv))
                               
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i, l in enumerate(self.layers):
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout*i)
            x = l(x, edge_index)
            if self.activation is not None:
                x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x, data.batch)
        x = self.ffnet(x)
        if self.softmax: 
            return F.log_softmax(x, dim=1)      
        else: 
            return x
    

