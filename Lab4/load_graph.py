import random
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PPIDataset

class simple_dataloader:
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.len = len(dataset)

    def __len__(self):
        return self.len


    def __iter__(self):
        self.n = 0
        self._order = [i for i in range(self.len)]
        if self.shuffle:
            random.shuffle(self._order)

        return self

    def __next__(self):
        if self.n >= self.len:
            raise StopIteration

        ret = self.dataset[self._order[self.n]]
        self.n += 1
        return ret


def Load_graph(data : str, task_type='node'):

    if task_type != 'node' and task_type != 'edge':
        raise Exception('error task_type')

    if data == 'cora':
        dataset = CoraGraphDataset()

    elif data == 'citeseer':
        dataset = CiteseerGraphDataset()

    elif data == 'ppi':
        dataset = PPIDataset()

    else:
        raise Exception('unknown data')

    return dataset