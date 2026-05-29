import torch

class Metric:
    '''
    Base class

    Provides interface:
    - reset
    - update
    - result
    '''
    def __init__(self):
        self.name = 'unnamed_metric'

    def reset(self):
        '''
        Called at the start of each epoch.
        '''
        pass

    def update(self, targets, outputs):
        '''
        Called for every batch of data.

        :param targets:
        :param outputs:
        '''
        pass

    def result(self):
        '''
        Called at the end of every epoch.
        '''
        return None


class Loss(Metric):
    def __init__(self):
        self.name = 'loss'

    def reset(self):
        self.total_loss = 0
        self.total_steps = 0

    def update(self, loss):
        self.total_loss += loss
        self.total_steps += 1

    def result(self):
        return self.total_loss / self.total_steps

