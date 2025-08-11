import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.learning import Learn

from model import EncoderLoss

class Learn(Learn):
    """Small variation from cosmosis.learning.Learn.run() to enable use of Encoderloss
    """
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
                                     num_workers=self.num_workers, pin_memory=True, 
                                     drop_last=drop_last)
        # tertiary loop
        for data in dataloader:
            if self.gpu: # overwrite the datadic with a new copy on the gpu
                if type(data) == dict: 
                    _data = {}
                    for k in data:
                        _data[k] = data[k].to('cuda:0', non_blocking=True)
                    data = _data
                else: 
                    data = data.to('cuda:0', non_blocking=True)

            y_pred = self.model(data)
            
            if flag == 'infer':
                self.metrics.predictions.append(y_pred)
                y = None
            else:
                # variation from cosmosis.learning.Learn()
                if self.target != None:
                    if type(data) == dict: y = data[self.target]
                    else: y = getattr(data, self.target)
                    
                self.opt.zero_grad()
                if isinstance(self.criterion, EncoderLoss):
                    b_loss, y_pred, y = self.criterion(*y_pred, data, flag)
                else:
                    b_loss = self.criterion(y_pred, y)
                    
                self.metrics.e_loss += b_loss.item()
                self.metrics.n += self.bs
                
                if self.metrics.metric_func is not None: 
                    self.metrics.y.append(y)
                    self.metrics.y_pred.append(y_pred)
                    
                if flag == 'train':
                    b_loss.backward()
                    self.opt.step()
                    if hasattr(self.criterion, 'discriminator'):
                        self.criterion.discriminator.reset_parameters()
                # end of variation
        if flag == 'val': 
            self.scheduler.step(self.metrics.e_loss)
            self.metrics.lr_log.append(self.opt.param_groups[0]['lr'])
            
        self.metrics.metric(flag)
        self.metrics.loss(flag)
        self.metrics.report(y_pred, y, flag)
        self.metrics.reset_loop()
                
