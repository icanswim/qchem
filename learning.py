from datetime import datetime
import logging
import random
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import no_grad, save, load, from_numpy, cat, as_tensor, compile, permute, arange, int64
from torch.utils.data import Sampler, DataLoader
from torch.nn import functional as F

from torcheval.metrics import functional as t_metrics

from sklearn import metrics as sk_metrics


class Metrics():
    """
    """
    def __init__(self, report_interval=10, metric_name=None, 
                     log_plot=False, min_lr=.00125, metric_param={}):
        
        now = datetime.now()
        self.start = now
        self.report_time = now
        self.report_interval = report_interval
        self.log_plot = log_plot
        self.min_lr = min_lr
        
        self.epoch, self.e_loss, self.n = 0, 0, 0
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.predictions, self.lr_log = [], []
        
        self.metric_name, self.metric_param = metric_name, metric_param
        self.metric_func, self.metric_train_log, self.metric_val_log = None, [], []
        self.y, self.y_pred = [], []
        
        if self.metric_name is not None:
            if self.metric_name in ['transformer']:
                self.metric_func = None
            elif self.metric_name in ['accuracy_score','roc_auc_score']:
                self.metric_func = getattr(sk_metrics, self.metric_name)
            elif self.metric_name in ['auc','multiclass_accuracy','multiclass_auprc']:
                self.metric_func = getattr(t_metrics, self.metric_name)
            else:
                raise Exception('hey just what you see pal...')
                
        logging.basicConfig(filename='./logs/cosmosis.log', level=20)
        self.log('\n.....................\n')
        self.log('\nNew Experiment: {}'.format(self.start))
    
    def infer(self):
        """
        process the predictions and save
        """
        now = datetime.now()
        print('\n.....................\n')
        self.log('\n.....................\n')
        self.log('\ninference job: {} \n'.format(self.start))
        self.log('\ntotal learning time: {} \n'.format(now - self.start))
        print('total learning time: {}'.format(now - self.start))
        
        if self.metric_name == 'transformer':
            predictions = F.softmax(self.predictions[-1].squeeze(), dim=-1)
            predictions = predictions.argmax(dim=-1)
            predictions = predictions.detach().cpu().numpy().tolist()
            predictions = self.decoder(predictions)
            predictions = np.asarray(predictions).reshape((1,-1))
            print('predictions: ', predictions)
        else:
            predictions = self.predictions.detach().cpu().numpy()
            print('predictions[-1]: ', predictions[-1])
            print('predictions.shape: ', predictions.shape)
            
        pd.DataFrame(predictions).to_csv(
                    './logs/{}_inference.csv'.format(now), index=True)
        print('inference instance {} complete and saved to csv...'.format(now))
        self.log('\ninference instance {} saved to csv...'.format(now))
        self.predictions = []
        
    def softmax_overflow(x):
        x_max = x.max(axis=1, keepdims=True)
        normalized = np.exp(x - x_max)
        return normalized / normalized.sum(axis=1, keepdims=True)
        
    def metric(self, flag):
        """
        called at the end of each run() loop
        TODO multiple metric
        flags = train, val, test, infer
        """
        if self.metric_func == None:
            return
            
        y_pred = cat(self.y_pred, dim=0)
        y = cat(self.y, dim=0)
        
        if self.metric_name in ['roc_auc_score','auc','accuracy_score',
                                    'multiclass_accuracy','multiclass_auprc']:
            y_pred = F.softmax(y_pred, dim=-1)

        if self.metric_name in ['accuracy_score','auc','multiclass_accuracy']:
            y_pred = y_pred.argmax(dim=-1)

        # sklearn metrics
        if self.metric_name in ['accuracy_score','roc_auc_score']:
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

        score = self.metric_func(y_pred, y, **self.metric_param) 
        score = score.item()
        
        if flag == 'train':
            self.metric_train_log.append(score)
        else:
            self.metric_val_log.append(score)
        
    def log(self, message):
        logging.info(message)
        
    def report(self, _y_pred, _y, flag):
        """
        called at the end of each run() loop
        """
        if flag == 'train': return
            
        now = datetime.now()
        elapsed = now - self.report_time
        if elapsed.total_seconds() < self.report_interval: return
            
        tot_elapsed = now - self.start
        print('\n.....................\n')
        print('total elapsed time: {}'.format(tot_elapsed))
        print('epoch: {}'.format(self.epoch))
        self.report_time = now

        if len(self.predictions) > 0: 
            print('len(self.predictions): ', len(self.predictions))
            return

        # get the last instance
        y_pred = _y_pred[-1]
        y = _y[-1]
        
        if self.metric_name == 'transformer':
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred = y_pred.argmax(dim=0)
            y_pred = y_pred.detach().cpu().numpy().tolist()
            y_pred = self.decoder(y_pred)
            y = y.detach().cpu().numpy().tolist()
            y = self.decoder(y)
            
        print('y_pred: ', y_pred)
        print('y: ', y)
        print('train loss: {}, val loss: {}'.format(self.train_loss[-1], self.val_loss[-1]))
        print('lr: {}'.format(self.lr_log[-1]))
    
        if len(self.metric_train_log) != 0:
            print('{} train score: {}, validation score: {}'.format(
                self.metric_name, self.metric_train_log[-1], self.metric_val_log[-1]))
    
    def loss(self, flag):
        """
        called at the end of each run() loop
        """
        if flag == 'train':
            self.train_loss.append(self.e_loss/self.n)
        if flag == 'val':
            self.val_loss.append(self.e_loss/self.n)
        if flag == 'test':
            self.test_loss.append(self.e_loss/self.n)
            
    def reset_loop(self):
        """
        called at the end of each run() loop
        """
        self.n, self.e_loss = 0, 0
        self.y, self.y_pred = [], []

    def final(self):
        now = datetime.now()
        print('\n........final........\n')
        self.log('\n........final........\n')
        self.log('\ntotal learning time: {}'.format(now - self.start))
        print('total learning time: {}'.format(now - self.start))
        
        if len(self.test_loss) != 0:
            self.log('test loss: {}'.format(self.test_loss))
            print('test loss: {}'.format(self.test_loss[-1]))
            
        if len(self.metric_train_log) != 0:
            self.log('\n{} test metric: {}'.format(self.metric_name, self.metric_val_log[-1]))
            print('{} test metric: {}'.format(self.metric_name, self.metric_val_log[-1]))
            logs = zip(self.train_loss, self.val_loss, self.lr_log, self.metric_val_log)
            cols = ['train_loss','validation_loss','learning_rate',self.metric_name]
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
    
    adapt = (D_in, D_out, dropout)
        prepends a trainable linear layer

    weights_only = True/False
        enable un pickling of models = False (only unpickle trusted files)
        
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
                 adapt=None, load_model=None, load_embed=True, save_model=False,
                 batch_size=10, epochs=1, compile_model=False, 
                 gpu=True, weights_only=False, target='y'):

        self.weights_only = weights_only
        self.gpu = gpu
        self.bs = batch_size
        self.target = target
        self.ds_param = ds_param
        self.dataset_manager(Datasets, Sampler, ds_param, sample_param)
        self.DataLoader = DataLoader
        
        self.metrics = Metrics(**metrics_param)
        if hasattr(self.train_ds, 'encoding'):
            self.metrics.decoder = self.train_ds.encoding.decode
        
        self.metrics.log('\nmodel: {}\n{}'.format(Model, model_param))
        self.metrics.log('\ndataset: {}\n{}'.format(Datasets, ds_param))
        self.metrics.log('\nsampler: {}\n{}'.format(Sampler, sample_param))
        self.metrics.log('\nepochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                                        epochs, batch_size, save_model, load_model))

        if not gpu: model_param['device'] = 'cpu'
        
        if load_model is not None:
            try: 
                model = Model(model_param)
                model.load_state_dict(load('./models/'+load_model, weights_only=self.weights_only))
                print('model loaded from state_dict...')
            except:
                model = load('./models/'+load_model, weights_only=self.weights_only)
                print('model loaded from pickle...')                                                      
        else:
            model = Model(model_param)
        
        if load_embed is True:
            try:
                for feature, embedding in model.embedding_layer.items():
                    weight = np.load('./models/{}_{}_embedding_weight.npy'.format(load_model[:-4], feature))
                    embedding.from_pretrained(from_numpy(weight), freeze=model_param['embed_param'][feature][3])
                print('loading embedding weights...')
            except:
                print('embedding weights failed to load.  reinitializing...')
                
        if adapt is not None: model.adapt(*adapt)

        if self.gpu == True:
            try:
                model.to('cuda:0')
                print('running model on gpu...')
            except:
                print('gpu not available.  running model on cpu...')
                self.gpu = False
        else:
            print('running model on cpu...')

        if compile_model:
            model = compile(model)
            print('compiling model...')
            
        self.model = model

        self.metrics.log('\n{}'.format(self.model.children))
        # primary loop 
        if Criterion is not None:
            self.criterion = Criterion(**crit_param)
            if self.gpu: self.criterion.to('cuda:0')
            self.metrics.log('\ncriterion: {}\n{}'.format(self.criterion, crit_param))
            self.opt = Optimizer(self.model.parameters(), **opt_param)
            self.metrics.log('\noptimizer: {}\n{}'.format(self.opt, opt_param))
            self.scheduler = Scheduler(self.opt, **sched_param)
            self.metrics.log('\nscheduler: {}\n{}'.format(self.scheduler, sched_param))
            
            for e in range(epochs):
                self.metrics.epoch = e
                self.sampler.shuffle_train_val_idx()
                self.run('train')
                with no_grad():
                    self.run('val')
                    if e > 1 and self.metrics.lr_log[-1] <= self.metrics.min_lr:
                        print('early stopping!  learning rate is below the set minimum...')
                        break
                
            with no_grad():
                self.run('test')
                
            self.metrics.final()
            
        else: # no Criterion implies inference mode
            with no_grad():
                for e in range(epochs): 
                    self.run('infer')
                    self.metrics.infer()
                    
        if save_model:
            if type(save_model) == str:
                model_name = save_model
            else:
                model_name = self.metrics.start.strftime("%Y%m%d_%H%M")

            if compile: 
                save(model.state_dict(), './models/{}.pth'.format(model_name))
            elif adapt: 
                save(self.model, './models/{}.pth'.format(model_name))
            else: 
                save(self.model.state_dict(), './models/{}.pth'.format(model_name))
                     
            if hasattr(self.model, 'embedding_layer'):
                for feature, embedding in self.model.embedding_layer.items():
                    weight = embedding.weight.detach().cpu().numpy()
                    np.save('./models/{}_{}_embedding_weight.npy'.format(model_name, feature), weight)

            print('model: {} saved...'.format(model_name))
    
    def run(self, flag): # secondary loop
        
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
            self.model.generate = True
            
        dataloader = self.DataLoader(dataset, batch_size=self.bs, 
                                     sampler=self.sampler(flag=flag), 
                                     num_workers=0, pin_memory=True, 
                                     drop_last=drop_last)
        # tertiary loop
        for data in dataloader:
            if self.gpu: # overwrite the datadic with a new copy on the gpu
                if type(data) == dict: 
                    _data = {}
                    for k, v in data.items():
                        _data[k] = data[k].to('cuda:0', non_blocking=True)
                    data = _data
                else: 
                    data = data.to('cuda:0', non_blocking=True)

            y_pred = self.model(data)   
            if self.metrics.metric_func is not None: self.metrics.y_pred.append(y_pred)
            
            if flag != 'infer':
                if type(data) == dict: y = data[self.target]
                else: y = getattr(data, self.target)
                    
                if self.metrics.metric_func is not None: self.metrics.y.append(y)
                    
                self.opt.zero_grad()
                b_loss = self.criterion(y_pred, y)
                self.metrics.e_loss += b_loss.item()
                self.metrics.n += self.bs

                if flag == 'train':
                    b_loss.backward()
                    self.opt.step()
            else:
                self.metrics.predictions.append(y_pred)
                y = None

        if flag == 'val': 
            self.scheduler.step(self.metrics.e_loss)
            self.metrics.lr_log.append(self.opt.param_groups[0]['lr'])
            
        self.metrics.metric(flag)
        self.metrics.loss(flag)
        self.metrics.report(y_pred, y, flag)
        self.metrics.reset_loop()
                
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


        
        
