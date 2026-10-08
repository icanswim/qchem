import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.learning import Learn

from model import EncoderLoss

class Learn(Learn):
    """Small variation from cosmosis.learning.Learn.run() to enable use of Encoderloss
    """
    def run(self, flag, prompt=False):

        if flag == 'train': 
            dataset = self.train_ds
            self.model.train(True)

        elif flag == 'val':   
            dataset = self.val_ds
            self.model.train(False)

        elif flag == 'test':  
            dataset = self.test_ds
            self.model.train(False)
 
        elif flag == 'infer':
            self.model.train(False)
            if prompt: 
                self.test_ds.ds = self.test_ds.prompt(prompt)
            dataset = self.test_ds
            self.model.generate = True
        dataloader = self.DataLoader(dataset, batch_size=self.bs, 
                                     sampler=self.sampler(flag), 
                                     num_workers=self.num_workers, 
                                     pin_memory=self.gpu, 
                                     drop_last=(flag != 'infer'))

        for data in dataloader:
            if isinstance(data, dict):
                data = {k: v.to(self.device, non_blocking=self.gpu) if hasattr(v, 'to') else v for k, v in data.items()}
                y = data[self.target] if flag != 'infer' else None  
            else:
                data = data.to(self.device, non_blocking=self.gpu)
                y = getattr(data, self.target) if flag != 'infer' else None

            y_pred = self.model(data)
            if flag == 'infer':
                self.metric.predictions.append(y_pred)
                return
            # variation from cosmosis.learning.Learn.run() 
            if isinstance(self.criterion, EncoderLoss):
                loss, y_pred, y = self.criterion(*y_pred, data, flag)
            else:
                loss = self.criterion(y_pred, y)

            if flag == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # variation from cosmosis.learning.Learn.run()
                if hasattr(self.criterion, 'discriminator'):
                    self.criterion.discriminator.reset_parameters()

            self.metric.e_loss += loss.item()
            self.metric.n += self.bs
            if self.metric.metric_func:
                self.metric.y.append(y)
                self.metric.y_pred.append(y_pred)
            
        if flag == 'val' and self.scheduler:
            self.scheduler.step(self.metric.e_loss)
            self.metric.lr.append(self.opt.param_groups[0]['lr'])
            
        self.metric.metric(flag)
        self.metric.loss(flag)
        self.metric.report(y_pred, y, flag)
        self.metric.reset_loop()
