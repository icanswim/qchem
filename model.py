import sys # required for relative imports in jupyter lab
import inspect
sys.path.insert(0, '../')

from cosmosis.model import CModel, FFNet

from torch import nn, log, mean, sum, exp, randn_like, matmul, Tensor, cat, sigmoid
from torch import isnan, log, abs, reshape

import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.nn import models as pygmodels
from torch_geometric.nn import aggr, conv, norm, pool
from torch_geometric.nn import NNConv, VGAE, GCNConv

from torch_geometric.utils import batched_negative_sampling, negative_sampling


class GModel(CModel):
    """Graph Model base"""

    def __init__(self, model_param):
        super().__init__(model_param)

        if 'pooling' in model_param and model_param['pooling'] is not None:
            self.pooling = getattr(pool, model_param['pooling'])
        else: 
            self.pooling = None

        if 'activation' in model_param and model_param['activation'] is not None:
            self.activation = getattr(nn, model_param['activation'])()
        else:
            self.activation = None
            
        if 'ffn_param' in model_param and model_param['ffn_param'] is not None:
            self.ffnet = FFNet(model_param['ffn_param'])
        else:
            self.ffnet = None

    def build(self, **kwargs):
        self.layers = []
        self.edge_attr = []
        self.data_keys = []
        raise NotImplementedError('subclass and implement build()...')

    def forward(self, data):
        """
        Data.feature = array
        """
        filter_keys = [] # keys not to be included or already included
        if self.y is not None: filter_keys.append(self.y)
        edge_attr = []
        x = []
        # if any features are to be embedded, embed them, add keys to filter
        if self.embed_param is not None:
            embedded_dict = self.embed_features(data)
            for e, embed in embedded_dict.items():
                if e in self.edge_attr:
                    edge_attr.append(embed)
                else:
                    x.append(embed)
                filter_keys.append(e)

        if len(self.edge_attr) > 0:
            for j in self.edge_attr:
                if j not in filter_keys:
                    edge_attr.append(getattr(data, j))
            edge_attr = cat(edge_attr, dim=1)
            
        if len(self.data_keys) > 0: 
            for k in self.data_keys:
                if k not in filter_keys:
                    x.append(getattr(data, k))
            x = cat(x, dim=-1)

        for l in self.layers: 
            if hasattr(l, 'forward'):
                # inspect the forward() of the layer inorder to route inputs
                fwd_param = inspect.signature(l.forward).parameters
                if 'input' in fwd_param: # the Sequence module
                    for s in l:
                        fwd_param = inspect.signature(s.forward).parameters
                        if 'x' and 'batch' in fwd_param: 
                            x = s(x, batch=data.batch)
                        elif 'x' and 'edge_index' and 'edge_attr' in fwd_param:
                            x = s(x, edge_index=data.edge_index, edge_attr=edge_attr)
                        elif 'x' and 'edge_index' in fwd_param:
                            x = s(x, edge_index=data.edge_index)
                        elif 'x' in fwd_param:
                            x = s(x)
                        else:
                            pass
                else:        
                    if 'x' and 'batch' in fwd_param:
                        x = l(x, batch=data.batch)
                    elif 'x' and 'edge_index' and 'edge_attr' in fwd_param:
                        x = l(x, edge_index=data.edge_index, edge_attr=edge_attr)
                    elif 'x' and 'edge_index' in fwd_param:
                        x = l(x, edge_index=data.edge_index)
                    elif 'x' in fwd_param:
                        x = l(x)
                    else:
                        pass  
                        
        if self.pooling is not None:
            x = self.pooling(x, batch=data.batch)
        if self.activation is not None:
            x = self.activation(x)
        if self.ffnet is not None:
            x = self.ffnet(x)
            
        return x

class PygModel(nn.Module):
    """
    A PyG model wrapper
    
    model_name = 'ModelName'
    pooling = 'global_mean' / None
    softmax = True/False
    pyg_param = {'in_channels': int,
                 'hidden_channels': int,
                 'num_layers': int,
                 'out_channels': int,
                 'dropout': float,
                 'norm': None}
    ffn_param = {'in_channels': int,
                 'hidden': int,
                 'out_channels': int,
                 'dropout': float,
                 'activation': str}
    """
    
    def __init__(self, model_param):
        super().__init__()
        
        launcher = getattr(pygmodels, model_param['model_name'])
        self.model = launcher(**model_param['pyg_param'])

        if 'pooling' in model_param and model_param['pooling'] is not None:
            self.pooling = getattr(pool, model_param['pooling'])
        else: self.pooling = None
            
        if 'ffn_param' in model_param and model_param['ffn_param'] is not None:
            self.ffnet = FFNet(model_param['ffn_param'])
        else: self.ffnet = None
            
        if 'softmax' in model_param and model_param['softmax'] is True:
            self.softmax = softmax
        else: self.softmax = False
        
        print('pytorch geometric model {} loaded...'.format(model_param['model_name']))
        
    def forward(self, data):

        x = self.model(data.x, data.edge_index)
        
        if self.pooling is not None:
            x = self.pooling(x, data.batch)
            
        if self.ffnet is not None:
            x = self.ffnet(x)  
            
        if self.softmax: 
            x = F.log_softmax(x, dim=1)
            
        return x
        
        
class NetConv(nn.Module):
    """NNConv wrapper which includes the network for edge attributes"""
    
    def __init__(self, in_channels, out_channels, edge_features=0, aggr='mean'):
        super().__init__()
        fnn = CModel.ff_unit(self, edge_features, in_channels*out_channels)
        self.conv = NNConv(in_channels, out_channels, fnn, aggr=aggr)

    def forward(self, x, edge_index, edge_attr):
        return self.conv.forward(x, edge_index, edge_attr)
        
          
class GraphNet(GModel):
    """
    builds PyG conv nets
    
    in_channels = node feature length
    out_channels = model output length
    hidden = hidden length
    depth = number of layers
    conv = 'SAGEConv'
    pooling = 'global_mean'/None
    dropout = float/None
    softmax = True/False
    activation = torch.nn.activation class name 'ReLU'
    normal = torch_geometric.norm class name 'LayerNorm'
    norm_param
    layer_param
    """
    def conv_unit(self, in_channels, out_channels, norm_param={}, layer_param={},
                      convolution='SAGEConv', dropout=.1, conv_act='ReLU', normal ='LayerNorm'):

        _conv=[]
        if convolution in ['NetConv']: 
            _conv.append(NetConv(in_channels, out_channels, **layer_param))
        else: 
            _conv.append(getattr(conv, convolution)(in_channels, out_channels, **layer_param))
        if normal is not None:
            _conv.append(getattr(norm, normal)(out_channels, **norm_param))
        if conv_act is not None: _conv.append(getattr(nn, conv_act)())
        if dropout is not None: _conv.append(nn.Dropout(p=dropout))
            
        return nn.Sequential(*_conv)
                  
    def build(self, in_channels=0, hidden=0, out_channels=0, depth=2, dropout=.2,
              pooling='global_mean_pool', activation='ReLU', normal='LayerNorm',
              convolution='SAGEConv', conv_act='ReLU', data_keys=['x'], edge_attr=[], 
              layer_param={}, norm_param={}, ffn_param=None, embed_param=None):

        """ffn_param={'in_channels': 0, 'hidden': 0, 'out_channels': 0, 'activation': 'ReLU'}
        """

        self.data_keys = data_keys
        self.edge_attr = edge_attr
        
        self.layers = []    
        self.layers.append(self.conv_unit(in_channels, hidden, convolution=convolution, 
                                          conv_act=conv_act, dropout=dropout, normal=normal,
                                          norm_param=norm_param, layer_param=layer_param))
        for d in range(depth-2):
            self.layers.append(self.conv_unit(hidden, hidden, convolution=convolution, 
                                          conv_act=conv_act, dropout=dropout, normal=normal,
                                          norm_param=norm_param, layer_param=layer_param))
        self.layers.append(self.conv_unit(hidden, out_channels, convolution=convolution, 
                                          conv_act=None, dropout=None, normal=None,
                                          norm_param=norm_param, layer_param=layer_param))
        
        if pooling is not None:
            self.pooling = getattr(pool, pooling)
        else: 
            self.pooling = None

        if activation is not None:
            self.activation = getattr(nn, activation)()
        else:
            self.activation = None
            
        if ffn_param is not None:
            self.ffnet = FFNet(ffn_param)
        else:
            self.ffnet = None

        print('GraphNet {} loaded...'.format(convolution))
                            
        
class GraphNetVariationalEncoder(GModel):
    """https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/nn/models/autoencoder.html
    https://arxiv.org/abs/1611.07308
    """
    
    def build(self, in_channels, hidden, out_channels, depth, 
                      convolution='GCNConv', pooling=None, **kwargs):
        
        self.gnet = GraphNet({'in_channels':in_channels, 'hidden':hidden, 'out_channels':hidden, 
                              'convolution':convolution, 'depth':depth, 'pooling':pooling, **kwargs})
        self.mu = self.ff_unit(hidden, hidden, activation=None, norm=False, dropout=None)
        self.logstd = self.ff_unit(hidden, hidden, activation=None, norm=False, dropout=None)
        print('GraphNetVariationalEncoder loaded...')

    def forward(self, data):
        z = self.gnet(data)
        mu = self.mu(z)
        logstd = self.logstd(z)

        #reparametrize
        if self.training:
            mu = mu + randn_like(logstd) * exp(logstd)

        return z, mu, logstd
        
        
class EncoderLoss():
    """criterion for Adversarial Variational Auto Encoders
    https://arxiv.org/abs/1802.04407
    
    Decoder = takes embeddings (z) and adjacency matrix (edge_index) returns 
        probabilities that an edge exists
        
    adversarial = True/False toggles adversarial regularizing MLP enforcing 
        standard normal distribution prior
    """
    
    def __init__(self, Decoder=pygmodels.InnerProductDecoder, decoder_param={},
                       adversarial=False, disc_param={}):
    
        self.decoder = Decoder(**decoder_param)
        self.adversarial = adversarial
        if self.adversarial:
            self.discriminator = FFNet(disc_param)
            self.disc_optimizer = Adam(self.discriminator.parameters(), lr=.01)
            
    def __call__(self, z, mu, logstd, data, flag):
        return self.forward(z, mu, logstd, data, flag)

    def to(self, device):
        self.decoder.to(device)
        if self.adversarial:
            self.discriminator.to(device)

    def reg_loss(self, z):
        """Computes the regularization loss of the encoder."""
        reg_loss = -log(sigmoid(self.discriminator(z)) + 1e-15).mean()
        return reg_loss

    def discriminator_loss(self, z):
        """Computes the loss of the discriminator"""
        real = sigmoid(self.discriminator(randn_like(z)))
        fake = sigmoid(self.discriminator(z.detach()))

        real_loss = -log(real + 1e-15).mean()
        fake_loss = -log(1 - fake + 1e-15).mean()

        return real_loss + fake_loss

    def recon_loss(self, z, data):
        """Given latent variable (z), computes the binary cross
        entropy loss for positive edges (pos_edge_index) and negative
        sampled edges (neg_edge_index)."""

        pos_edge_index = data.edge_index
        neg_edge_index = batched_negative_sampling(pos_edge_index, data.batch, 
                                                       method='dense', force_undirected=True)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)

        pos_loss = -log(pos_pred + 1e-15).mean()
        neg_loss = -log(1 - neg_pred + 1e-15).mean()
        recon_loss = pos_loss + neg_loss

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = cat([pos_y, neg_y], dim=0)
        y_pred = cat([pos_pred, neg_pred], dim=0)

        return recon_loss, y_pred, y

    def kl_loss(self, mu, logstd):
        logstd = logstd.clamp(max=10)
        return -0.5 * mean(sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
    def forward(self, z, mu, logstd, data, flag):
        recon_loss, y_pred, y = self.recon_loss(z, data)
        kl_loss = self.kl_loss(mu, logstd)

        if self.adversarial:
            if flag == 'train':
                for _ in range(5):
                    self.disc_optimizer.zero_grad()
                    disc_loss = self.discriminator_loss(z)
                    disc_loss.backward()
                    self.disc_optimizer.step()
            reg_loss = self.reg_loss(z)
            loss = recon_loss + kl_loss + reg_loss
        else: 
            loss = recon_loss + kl_loss

        return loss, y_pred, y
        
            
        
            
            
            
        
        


