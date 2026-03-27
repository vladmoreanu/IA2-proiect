# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from xor_dataset import create_xor_dataset

x, y = create_xor_dataset(num_samples_per_class=400, noise=0.25)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=33
    )

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

from torch import FloatTensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(
    FloatTensor(x_train), 
    LongTensor(y_train)
    )
test_dataset = TensorDataset(
    FloatTensor(x_test), 
    LongTensor(y_test)
    )

print("Train/Test elements: ", len(train_dataset), len(test_dataset))


# %%
import torch

import importlib
import lighter
importlib.reload(lighter)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'System :')
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version PyTorch expects: {torch.version.cuda}')
print(f'  Using device: {device}, {torch.device(device)}')

class Accuracy(lighter.metrics.Metric):
    def __init__(self):
        self.name = 'acc'

    def reset(self):
        self.correct_predictions = 0
        self.total_samples = 0

    def update(self, targets, outputs):
        _, predicted = torch.max(outputs, 1)
        self.correct_predictions += (predicted == targets).sum().item()
        self.total_samples += targets.size(0)

    def result(self):
        return (self.correct_predictions / self.total_samples) \
            if self.total_samples > 0 else None

class FFN(lighter.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.activation = torch.nn.Sigmoid()
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


input_size  = 2
hidden_size = 128
output_size = 2

epochs = 1000
learning_rate = 1e-3

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = FFN(input_size, hidden_size, output_size)

model.compile(
    torch.optim.Adam(model.parameters(), lr=learning_rate),
    torch.nn.CrossEntropyLoss(),
    metrics=[
        Accuracy()
    ]
)


train_l, test_l = model.fit(
    train_loader,
    epochs,
    validation_loader=val_loader,
    validation_freq=5,
    callbacks=[
        lighter.callbacks.History(),
        lighter.callbacks.Checkpoint(
            './checkpoints/xor_test_{epoch}.pt',
            save_best_only=True
        ),
    ]
)


# %%
# import importlib
# import lighter
# importlib.reload(lighter)

# lighter.utils.plot_decision_boundary(model, train_loader)
# lighter.utils.plot_decision_boundary(model, val_loader)
# lighter.utils.plot_loss(train_l, test_l, title='Model Losses')


# %%
# x_test_tensor = torch.FloatTensor(x_test)
# y_test_tensor = torch.FloatTensor(y_test)

# y_pred_tensor = model(x_test_tensor)

# yy, predicted = torch.max(y_pred_tensor, 1)

# (y_test_tensor == predicted).sum().item() / len(y_test_tensor)



