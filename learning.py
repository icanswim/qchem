from datetime import datetime
import logging
import random
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import no_grad, save, load, from_numpy, squeeze
from torch.utils.data import Sampler, DataLoader
from torch.nn.functional import softmax

from sklearn import metrics

from model import EncoderLoss


class Metrics():

    def __init__(self, report_interval=10, sk_metric_name=None, 
                     log_plot=False, min_lr=.00125, sk_param={}):
        
        self.start = datetime.now()
        self.report_time = self.start
        self.report_interval = report_interval
        self.log_plot = log_plot
        self.min_lr = min_lr
        
        self.epoch, self.e_loss, self.predictions = 0, [], []
        self.train_loss, self.val_loss, self.lr_log = [], [], []
        
        self.sk_metric_name, self.sk_param = sk_metric_name, sk_param
        self.skm, self.sk_train_log, self.sk_val_log = None, [], []
        self.y, self.last_y, self.y_pred, self.last_y_pred = [], [], [], []
        if self.sk_metric_name is not None:
            self.skm = getattr(metrics, self.sk_metric_name)
            
        logging.basicConfig(filename='./logs/cosmosis.log', level=20)
        self.log('\nNew Experiment: {}'.format(self.start))
    
    def infer(self):
        self.predictions = np.concatenate(self.predictions)
        self.predictions = pd.DataFrame(self.predictions)
        self.predictions.to_csv('./logs/{}_inference.csv'.format(self.start), index=True)
        print('inference {} complete and saved to csv...'.format(self.start))

    def metric(self, flag):
        """TODO multiple sk metrics"""
    
        def softmax(x): return np.exp(x)/sum(np.exp(x))

        def softmax_overflow(x):
            x_max = x.max()
            normalized = np.exp(x - x_max)
            return normalized / normalized.sum()

        y = np.concatenate(self.y)
        y_pred = np.concatenate(self.y_pred)
        
        if self.sk_metric_name is not None:
            if self.sk_metric_name == 'roc_auc_score' and y_pred.ndim == 2:
                y_pred = np.apply_along_axis(softmax_overflow, 1, y_pred)
            
            if self.sk_metric_name == 'accuracy_score' and y_pred.ndim == 2:
                y_pred = np.argmax(y_pred, axis=1)

            score = self.skm(y, y_pred, **self.sk_param)
            if flag == 'train':
                self.sk_train_log.append(score)
            else:
                self.sk_val_log.append(score)

        self.last_y, self.last_y_pred = np.squeeze(y[-5:]), np.squeeze(y_pred[-5:])
        self.y, self.y_pred = [], []
        
    def loss(self, flag, loss):
        if flag == 'train':
            self.train_loss.append(loss)
        if flag == 'val':
            self.val_loss.append(loss)
        if flag == 'test':
            self.log('test loss: {}'.format(loss))
            print('test loss: {}'.format(loss))
          
    def log(self, message):
        logging.info(message)
        
    def status_report(self, now=False):
        
        def print_report():
            print('\n...........................')
            print('learning time: {}'.format(datetime.now()-self.start))
            print('epoch: {}, lr: {}'.format(self.epoch, self.lr_log[-1]))
            print('train loss: {}, val loss: {}'.format(self.train_loss[-1], self.val_loss[-1]))
            print('last 5 targets: \n{}'.format(self.last_y))
            print('last 5 predictions: \n{}'.format(self.last_y_pred))
            if self.skm is not None:
                print('sklearn train metric: {}, sklearn validation metric: {}'.format(
                                                    self.sk_train_log[-1], self.sk_val_log[-1]))
            self.report_time = datetime.now()
        
        if now:
            print_report()
        else:
            elapsed = datetime.now() - self.report_time
            if elapsed.total_seconds() > self.report_interval or self.epoch % 10 == 0:
                print_report()
        
    def final_report(self):
        elapsed = datetime.now() - self.start
        print('\n...........................')
        self.log('learning time: {} \n'.format(elapsed))
        print('learning time: {}'.format(elapsed))
        print('last 5 targets: \n{}'.format(self.last_y))
        print('last 5 predictions: \n{}'.format(self.last_y_pred))
        
        if self.skm is not None:
            self.log('sklearn test metric: \n{} \n'.format(self.sk_val_log[-1]))
            print('sklearn test metric: \n{} \n'.format(self.sk_val_log[-1]))
            logs = zip(self.train_loss, self.val_loss, self.lr_log, self.sk_val_log)
            cols = ['train_loss','validation_loss','learning_rate',self.sk_metric_name]
        else:
            logs = zip(self.train_loss, self.val_loss, self.lr_log)
            cols = ['train_loss','validation_loss','learning_rate']
            
        pd.DataFrame(logs, columns=cols).to_csv('./logs/'+self.start.strftime("%Y%m%d_%H%M"))
        self.view_log('./logs/'+self.start.strftime('%Y%m%d_%H%M'), self.log_plot)
        
    @classmethod    
    def view_log(cls, log_file, log_plot):
        log = pd.read_csv(log_file)
        log.iloc[:,1:5].plot(logy=log_plot)
        plt.show() 


class Selector(Sampler):
    """splits = (train_split,) remainder is val_split or 
                (train_split,val_split) remainder is test_split or None
    """
    def __init__(self, dataset_idx=None, train_idx=None, val_idx=None, test_idx=None,
                 splits=(.7,.15), set_seed=False, subset=False):
        self.set_seed = set_seed
        
        if dataset_idx == None:  
            self.dataset_idx = train_idx
        else:
            self.dataset_idx = dataset_idx
            
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        
        if set_seed: 
            random.seed(set_seed)
            
        random.shuffle(self.dataset_idx) 
        if subset:
            sub = int(len(self.dataset_idx)*subset)
            self.dataset_idx = self.dataset_idx[:sub]
            
        if len(splits) == 1:  
            cut1 = int(len(self.dataset_idx)*splits[0])
            self.train_idx = self.dataset_idx[:cut1]
            self.val_idx = self.dataset_idx[cut1:]
        if len(splits) == 2:
            cut1 = int(len(self.dataset_idx)*splits[0])
            cut2 = int(len(self.dataset_idx)*splits[1])
            self.train_idx = self.dataset_idx[:cut1]
            self.val_idx = self.dataset_idx[cut1:cut1+cut2]
            self.test_idx = self.dataset_idx[cut1+cut2:]
        
        random.seed()
        
    def __iter__(self):
        if self.flag == 'train':
            return iter(self.train_idx)
        if self.flag == 'val':
            return iter(self.val_idx)
        if self.flag == 'test':
            return iter(self.test_idx)
        if self.flag == 'infer':
            return iter(self.dataset_idx)

    def __len__(self):
        if self.flag == 'train':
            return len(self.train_idx)
        if self.flag == 'val':
            return len(self.val_idx)
        if self.flag == 'test':
            return len(self.test_idx) 
        if self.flag == 'infer':
            return len(self.dataset_idx)
        
    def __call__(self, flag):
        self.flag = flag
        return self
    
    def shuffle_train_val_idx(self):
        if self.set_seed:
            random.seed(self.set_seed)
        random.shuffle(self.val_idx)
        random.shuffle(self.train_idx)
        random.seed()
        
        
class Learn():
    """
    Datasets = [TrainDS, ValDS, TestDS]
        if 1 DS is given it is split into train/val/test using splits param
        if 2 DS are given first one is train/val second is test
        if 3 DS are given first is train second is val third is test
        
    Criterion = None implies inference mode
    
    load_model = None/'saved_model.pth'/'saved_model.pk'
    load_embed = None/'model_name'
    squeeze_y_pred = True/False (torch.squeeze(y_pred)) 
        squeeze the model output
    adapt = (D_in, D_out, dropout)
        prepends a trainable linear layer
        
    the dataset output can either be a dictionary utilizing the form 
    data = {'model_input': {},
            'criterion_input': {'target':{}}} 
    or an object with a feature 'target' (data.target)
    the entire data object is passed to the model
    """
    def __init__(self, Datasets, Model, Sampler=Selector, Metrics=Metrics,
                 DataLoader=DataLoader,
                 Optimizer=None, Scheduler=None, Criterion=None, 
                 ds_param={}, model_param={}, sample_param={},
                 opt_param={}, sched_param={}, crit_param={}, metrics_param={}, 
                 adapt=None, load_model=None, load_embed=None, save_model=False,
                 batch_size=10, epochs=1, squeeze_y_pred=False, gpu=True, target='y'):
        
        self.gpu = gpu
        self.bs = batch_size
        self.squeeze_y_pred = squeeze_y_pred
        self.target = target
        self.ds_param = ds_param
        self.dataset_manager(Datasets, Sampler, ds_param, sample_param)
        self.DataLoader = DataLoader
        
        self.metrics = Metrics(**metrics_param)
        self.metrics.log('model: {}\n{}'.format(Model, model_param))
        self.metrics.log('dataset: {}\n{}'.format(Datasets, ds_param))
        self.metrics.log('sampler: {}\n{}'.format(Sampler, sample_param))
        self.metrics.log('epochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                                    epochs, batch_size, save_model, load_model))

        if not gpu: model_param['device'] = 'cpu'
        
        if load_model is not None:
            try: 
                model = Model(model_param)
                model.load_state_dict(load('./models/'+load_model))
                print('model loaded from state_dict...')
            except:
                model = load('./models/'+load_model)
                print('model loaded from pickle...')                                                      
        else:
            model = Model(model_param)
        
        if load_embed is not None:
            for i, embedding in enumerate(model.embeddings):
                weight = np.load('./models/{}_{}_embedding_weight.npy'.format(load_embed, i))
                embedding.from_pretrained(from_numpy(weight), freeze=model_param['embed_param'][i][4])
            print('loading embedding weights...')
          
        if adapt is not None: model.adapt(*adapt)
        
        if self.gpu:
            try:
                self.model = model.to('cuda:0')
                print('running model on gpu...')
            except:
                print('gpu not available.  running model on cpu...')
                self.model = model
                self.gpu = False
        else:
            print('running model on cpu...')
            self.model = model
            
        self.metrics.log(self.model.children)
        
        if Criterion is not None:
            self.criterion = Criterion(**crit_param)
            if self.gpu: 
                if isinstance(self.criterion, EncoderLoss):
                    self.criterion.decoder.to('cuda:0')
                    if self.criterion.adversarial:
                        self.criterion.discriminator.to('cuda:0')
                else: 
                    self.criterion.to('cuda:0')
            self.metrics.log('criterion: {}\n{}'.format(self.criterion, crit_param))
            self.opt = Optimizer(self.model.parameters(), **opt_param)
            self.metrics.log('optimizer: {}\n{}'.format(self.opt, opt_param))
            self.scheduler = Scheduler(self.opt, **sched_param)
            self.metrics.log('scheduler: {}\n{}'.format(self.scheduler, sched_param))
            
            for e in range(epochs): 
                self.metrics.epoch = e
                self.sampler.shuffle_train_val_idx()
                self.run('train')
                with no_grad():
                    self.run('val')
                    if e > 1 and  self.metrics.lr_log[-1] <= self.metrics.min_lr:
                        self.metrics.status_report(now=True)
                        print('early stopping!  learning rate is below the set minimum...')
                        break
                
            with no_grad():
                self.run('test')
                
            self.metrics.final_report()  
            
        else: #no Criterion implies inference mode
            with no_grad():
                self.run('infer')
        
        if save_model:
            if type(save_model) == str:
                model_name = save_model
            else:
                model_name = self.metrics.start.strftime("%Y%m%d_%H%M")
            if adapt: 
                save(self.model, './models/{}.pth'.format(model_name))
            else: 
                save(self.model.state_dict(), './models/{}.pth'.format(model_name))
                     
            if hasattr(self.model, 'embeddings'):
                for i, embedding in enumerate(self.model.embeddings):
                    weight = embedding.weight.detach().cpu().numpy()
                    np.save('./models/{}_{}_embedding_weight.npy'.format(model_name, i), weight)
        
    def run(self, flag): 
        e_loss, e_sk, i = 0, 0, 0
        if flag == 'train': 
            self.model.training = True
            dataset = self.train_ds
            drop_last = True
            
        if flag == 'val':
            self.model.training = False
            dataset = self.val_ds
            drop_last = True

        if flag == 'test':
            self.model.training = False
            dataset = self.test_ds
            drop_last = True
            
        if flag == 'infer':
            self.model.training = False
            dataset = self.test_ds
            drop_last = False

        dataloader = self.DataLoader(dataset, batch_size=self.bs, 
                                     sampler=self.sampler(flag=flag), 
                                     num_workers=0, pin_memory=True, 
                                     drop_last=drop_last)
       
        for data in dataloader:
            i += self.bs
            if self.gpu: # overwrite the datadic with a new copy on the gpu
                if type(data) == dict: 
                    _data = {}
                    for k, v in data.items():
                        _data[k] = data[k].to('cuda:0', non_blocking=True)
                    data = _data
                else: 
                    data = data.to('cuda:0', non_blocking=True)
            y_pred = self.model(data)
     
            if self.squeeze_y_pred: y_pred = squeeze(y_pred)
                
            if flag == 'infer':
                self.metrics.predictions.append(y_pred.detach().cpu().numpy())
            else:
                if type(data) == dict:
                    y = data[self.target]
                else: 
                    y = getattr(data, self.target)
                self.opt.zero_grad()
                #variation from Cosmo
                if isinstance(self.criterion, EncoderLoss):
                    b_loss, y_pred, y = self.criterion(*y_pred, data, flag)
                else:
                    b_loss = self.criterion(y_pred, y)
                e_loss += b_loss.item()
                
                self.metrics.y.append(y.detach().cpu().numpy())
                self.metrics.y_pred.append(y_pred.detach().cpu().numpy())
                
                if flag == 'train':
                    b_loss.backward()
                    self.opt.step()
                    
        if flag == 'infer':
            self.metrics.infer()
        else:
            self.metrics.loss(flag, e_loss/i)
            self.metrics.metric(flag)
            
        if flag == 'val': 
            self.scheduler.step(e_loss/i)
            self.metrics.lr_log.append(self.opt.param_groups[0]['lr'])
            self.metrics.status_report()
            
    def dataset_manager(self, Datasets, Sampler, ds_param, sample_param):
    
        if len(Datasets) == 1:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = self.test_ds = self.train_ds
            self.sampler = Sampler(dataset_idx=self.train_ds.ds_idx, 
                                   **sample_param)
        if len(Datasets) == 2:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = self.train_ds
            self.test_ds = Datasets[1](**ds_param['test_param'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                   test_idx=self.test_ds.ds_idx,
                                   **sample_param)
        if len(Datasets) == 3:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = Datasets[1](**ds_param['val_param'])
            self.test_ds = Datasets[2](**ds_param['test_param'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                   val_idx=self.val_ds.ds_idx, 
                                   test_idx=self.test_ds.ds_idx,
                                   **sample_param)



        
        
