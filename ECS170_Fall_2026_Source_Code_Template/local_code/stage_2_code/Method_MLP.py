# Stage 2 MLP w/ adjustable layers for classifying handwritten digits (784 pixels -> 10 digits).
# TODO: clean up this file later, kinda messy right now lol

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # experiment knobs, tweak these on the object b4 calling run()
    max_epoch = 20
    batch_size = 128
    learning_rate = 1e-3
    optimizer_class = torch.optim.Adam
    optimizer_kwargs = {}
    # 'cpu', 'cuda' or 'mps' (mps = ur Mac's GPU)
    device = 'cpu'
    # set False to hide the per-epoch lines when running a big experiment sweep
    verbose = True

    def __init__(self, mName, mDescription, input_dim=784, hidden_dims=(256, 128), output_dim=10,
                 activation=nn.ReLU, dropout=0.0, loss_function=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.optimizer_kwargs = dict(self.optimizer_kwargs)

        # collect the layers in a list first, then hand them to Sequential below
        layers = []
        in_dim = input_dim

        # loop through hidden dims
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())

            # added dropout if we need to fight overfitting (remember to check if this actually helps later)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = h

        # output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.layers = nn.Sequential(*layers)

        # CrossEntropyLoss does softmax on its own, so forward() just gives raw logits.
        # Pass loss_function=... to swap it for one of Loss_Functions.build_loss();
        # every loss there also takes (logits, integer labels).
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
        self.loss_function = loss_function

        # per-epoch numbers for the convergence plots, val_* stays empty w/o data['val']
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def forward(self, x):
        # standard forward pass
        out = self.layers(x)
        return out

    def _tensors(self, X, y=None):
        # helper to turn lists/arrays into tensors w/ the right dtype and device
        X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)

        if y is None:
            return X_tensor

        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        return X_tensor, y_tensor

    @torch.no_grad()
    def _loss_and_acc(self, X, y):
        nn.Module.train(self, False) # eval mode
        logits = self.forward(X)
        loss_val = self.loss_function(logits, y).item()

        preds = logits.argmax(1)
        acc_val = (preds == y).float().mean().item()

        return loss_val, acc_val

    # called train_model not train bc nn.Module.eval() calls self.train(False)
    def train_model(self, X, y, X_val=None, y_val=None):
        if self.history['train_loss']:
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            self.history = {key: [] for key in self.history}

        self.to(self.device)
        X, y = self._tensors(X, y)

        if X_val is not None:
            X_val, y_val = self._tensors(X_val, y_val)

        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)

        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)

            # shuffle data every epoch
            perm = torch.randperm(len(X), device=self.device)

            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]

                batch_x = X[idx]
                batch_y = y[idx]

                optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = self.loss_function(logits, batch_y)

                loss.backward()
                optimizer.step()

            # evaluate after each epoch
            train_loss, train_acc = self._loss_and_acc(X, y)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            msg = f'Epoch: {epoch} Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}'

            if X_val is not None:
                val_loss, val_acc = self._loss_and_acc(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                msg += f' Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}'

            if self.verbose:
                print(msg)

    @torch.no_grad()
    def test(self, X):
        nn.Module.train(self, False)
        X_tensor = self._tensors(X)
        logits = self.forward(X_tensor)
        preds = logits.argmax(1).cpu()
        return preds

    def run(self):
        print('method running...')
        print('--start training...')

        val = self.data.get('val', {})
        val_x = val.get('X', None)
        val_y = val.get('y', None)

        self.train_model(self.data['train']['X'], self.data['train']['y'], val_x, val_y)

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y']
        }
