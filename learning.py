import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.learning import *


class ChampSelector(Selector):
    """This class is for use with the Champs dataset.  If the Champs dataset has been created as an 
    undirected graph with connections (scc) pointing in both directions (if atom_idx_0 points to atom_idx_1
    then atom_idx_1 also points atom_idx_0) then when selecting the test hold out set both directions 
    need to be selected inorder to prevent a data leak.
    TODO doesnt work for inference, use Selector class
    TODO try training a model on only one direction and testing on the opposite
    """
    def __init__(self, dataset_idx, split=.1, subset=False):
        self.split = split
        self.half = int(len(dataset_idx)/2) # 4658147
        first = dataset_idx[:self.half] # only sample from the first half; second half is the reverse connections

        if subset:
            dataset_idx = random.sample(first, int(len(first)*subset)) 
        else:    
            dataset_idx = first
        
        random.shuffle(dataset_idx)
        cut = int(len(dataset_idx)//(1/self.split))
        self.test_idx = dataset_idx[:cut]
        self.dataset_idx = dataset_idx[cut:]
        
        # add the reverse connections
        test_index = self.test_idx.copy()
        for i in test_index:
            self.test_idx.append(i+self.half)
            
    def sample_train_val_idx(self):
        
        cut = int(len(self.dataset_idx)*self.split)
        random.shuffle(self.dataset_idx)
        
        self.val_idx = self.dataset_idx[:cut]
        self.train_idx = self.dataset_idx[cut:]
        
        val_index = self.val_idx.copy()
        for i in val_index:
            self.val_idx.append(i+self.half)
     
        train_index = self.train_idx.copy()
        for i in val_index:
            self.train_idx.append(i+self.half)