import torch

class Metric:
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


class PSNR(Metric):
    def __init__(self):
        self.count = None
        self.total_psnr = None
        self.name = 'psnr'

    def reset(self):
        self.total_psnr = 0
        self.count = 0

    def update(self, targets, outputs):
        mse = torch.mean((targets - outputs) ** 2)
        
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # for pixels range [0, 1]

        self.total_psnr += psnr.item()
        self.count += 1

    def result(self):
        return self.total_psnr / self.count if self.count > 1 else 0


