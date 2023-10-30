import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.model import CModel, FFNet

from torch import nn, log, mean, sum, exp, randn_like
import torch.nn.functional as F


from torch_geometric.nn import models as pygmodels
from torch_geometric.nn import aggr, conv, NNConv, VGAE, GCNConv

from torch_geometric.utils import negative_sampling


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

        launcher = getattr(pygmodels, model_params['model_name'])
        self.model = launcher(**model_params['pyg_params'])
        
        pool = model_params['pool']
        if pool is not None:
            self.pool = getattr(aggr, pool)()
        else:
            self.pool = None
        
        self.ffnet = model_params['ffnet']
        if self.ffnet:
            self.ffn = FFNet({'in_channels': model_params['in_channels'], 
                              'hidden': model_params['hidden'],
                              'out_channels': model_params['out_channels']})
            
        self.softmax = model_params['softmax']
        
        print('pytorch geometric model {} loaded...'.format(model_params['model_name']))
        
    def forward(self, data):

        x = self.model(data.x, data.edge_index)
        
        if self.pool is not None:
            x = self.pool(x, data.batch)
            
        if self.ffnet: 
            x = self.ffn(x)  
            
        if self.softmax: 
            x = F.log_softmax(x, dim=1)
            
        return x
        
        
class NetConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, edge_features=0):
        super().__init__()
        
        nn = CModel.ff_unit(self, edge_features, in_channels*out_channels)
        self.conv = NNConv(in_channels, out_channels, nn, aggr='mean')
        
    def forward(self, x, edge_index, edge_attr):
        return self.conv.forward(x, edge_index, edge_attr)
    
          
class GraphNet(CModel):
    """
    builds PyG conv nets
    
    in_channels = node feature length
    out_channels = model output length
    hidden = hidden length
    depth = number of layers
    conv = 'SAGEConv'
    pool = 'global_mean'/None
    dropout = .int/None
    softmax = True/False
    activation = F.activation
    """

    def build(self, in_channels=0, hidden=0, out_channels=0, depth=0, 
              convolution='SAGEConv', pool='MeanAggregation', dropout=.1, 
              softmax=False, activation='relu', **kwargs):
        
        self.dropout = dropout
        self.softmax = softmax
        self.convolution = convolution
        
        if activation is not None:
            self.activation = getattr(F, activation)
        else: 
            self.activation = None    
        
        if pool is not None:
            self.pool = getattr(aggr, pool)()
        else:
            self.pool = None
            
        layers = []    
        if self.convolution == 'NetConv':
            Conv = NetConv
        else:
            Conv = getattr(conv, convolution)
            
        layers.append(Conv(in_channels, hidden, **kwargs))
        for d in range(depth):
            layers.append(Conv(hidden, hidden, **kwargs))    
        self.layers = layers
        
        self.ffnet = FFNet({'in_channels':hidden, 'hidden':2*hidden, 
                                            'out_channels':out_channels})
        
        print('GraphNet {} loaded...'.format(conv))
                               
    def forward(self, data):
        x = data.x
        
        for i, l in enumerate(self.layers):
            
            if self.convolution in ['NetConv']:
                x = l(x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            else:
                x = l(x, edge_index=data.edge_index)
            
            if self.activation is not None:
                x = F.relu(x)
                
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout*i)
                
        if self.pool is not None:
            x = self.pool(x, data.batch)
            
        x = self.ffnet(x)
        
        if self.softmax: 
            x = F.log_softmax(x, dim=1)      
        
        return x
    

    
class GraphNetVariationalEncoder(CModel):
    
    def build(self, in_channels, hidden, out_channels, depth):
        self.gnet = GraphNet({'in_channels':in_channels, 'hidden':2*hidden, 
                              'out_channels':hidden, 'depth': depth})
        self.mu = FFNet({'in_channels':hidden, 'hidden':2*hidden, 
                         'out_channels':out_channels})
        self.logstd = FFNet({'in_channels':hidden, 'hidden':2*hidden, 
                             'out_channels':out_channels})

    def forward(self, data):
        z = self.gnet(data)
        mu = self.mu(z)
        logstd = self.logstd(z)
        
        #reparametrize
        if self.training:
            z = mu + randn_like(logstd) * exp(logstd)
            return (z, mu, logstd)
        else:
            return (z, mu, logstd)
    

class QGAE(CModel):
    """Quantum Graph Auto Encoder"""

    def build(self, in_channels, hidden, out_channels, depth):
        self.encoder = GraphNetVariationalEncoder(in_channels, hidden, out_channels, depth)


    def forward(self, data):        
        
        if hasattr(self, 'training'):
            self.encoder.training = self.training
        (z, mu, logstd) = self.encoder(data)

    
class CriterionQGAE():
    """criterion for auto encoders"""
    
    def __init__(self, Decoder=pygmodels.InnerProductDecoder, decoder_params={}):
        self.decoder = Decoder(**decoder_params)

    def forward(self, (z, mu, std), pos_edge_index):
        
        pos_loss = -log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            
        neg_loss = -log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        recon_loss = pos_loss + neg_loss
        
        def kl_loss():
            return -0.5 * mean(sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
        loss = recon_loss + (1 / data.num_nodes) * k1_loss()
        return loss
    
