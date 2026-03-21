import torch
import torch.utils.data
from tqdm import tqdm

class Metric:
    def __init__(self):
        self.name = 'unnamed_metric'

    def reset(self):
        '''
        Called at the start of each training step.
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
        Called at the end of every training step.
        '''
        return None


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(
        self,
        x
    ):
        return x


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


    def compile(
        self,
        optimizer=None,
        loss=None,
        # loss_weights=None,
        metrics : list[Metric] = None,
        # weighted_metrics=None,
        # run_eagerly=False,
        # steps_per_execution=1,
        # jit_compile="auto",
        # auto_scale_loss=True,
        device='cpu',
    ):
        self.optimizer  = optimizer
        self.loss_fn    = loss
        self.metrics    = [self.Loss()] + metrics
        self.device     = device


    def step(
        self,
        data_loader : torch.utils.data.DataLoader,
        training : bool = False,
    ):
        """
        Single-iteration function, either for training or for testing.

        :param dataloader: torch.utils.data.DataLoader object to iterate over
        """
        self.train() if training else self.eval()

        for metric in self.metrics:
            metric.reset()

        for inputs, targets in data_loader: # tqdm(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            if training:
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self(inputs)
                    loss = self.loss_fn(outputs, targets) 

            # Metrics update state
            for metric in self.metrics:
                if metric.name == 'loss':
                    metric.update(loss.item())
                else:
                    metric.update(targets, outputs)


    def fit(
        self,
        train_loader=None,
        epochs=1,
        # verbose="auto",
        # batch_size=None,
        callbacks=None,
        validation_loader=None,
        # shuffle=True,
        # class_weight=None,
        # sample_weight=None,
        # initial_epoch=1,
        # steps_per_epoch=None,
        # validation_rate=None,
        # validation_batch_size=None,
        validation_freq=1,
    ):
        """
        Train loop.
        """
        train_losses = []
        val_losses = []
        
        # best_loss = torch.inf
        self.to(self.device)

        def report(training=False):
            prefix = 'train' if training else 'val'
            out_str = '\t'
            for metric in self.metrics:
                x = metric.result()
                x = f'{x:.3f}' if x > 0.01 else f'{x:.2e}'
                out_str += ' {:s}_{:s}={:s}'.format(prefix, metric.name, x)
            print(out_str)

        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # else:
        #     try:
        #         model.load_state_dict(
        #             torch.load(
        #                 os.path.join(folder_path, file_name)
        #             )["state_dict"]
        #         )
        #     except:
        #         print("Couldn't load model")
        
        for e in range(1, epochs + 1):
            print(f"Epoch {e}/{epochs}:")
            
            self.step(train_loader, training = True)
            train_losses.append(self.metrics[0].result())
            report(training=True)

            if (validation_loader is not None) & (e % validation_freq == 0):
                self.step(validation_loader)
                val_losses.append(self.metrics[0].result())
                report()
            
            # TODO: CALLBACKS HERE

            # if train_loss < best_loss:
            #     best_loss = train_loss
        
            #     checkpoint = {
            #         'state_dict': model.state_dict()
            #         'optimizer': optimizer.state_dict()
            #     }
            #     torch.save(checkpoint, os.path.join(folder_path, file_name))

        if validation_loader:
            return train_losses, val_losses
        else:
            return train_losses
        
