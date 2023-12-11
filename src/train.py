import torch
from torch import nn, from_numpy
from torch.utils.data import DataLoader

from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix, BinaryPrecision, BinaryRecall

from numpy import load as npload, float32
from pathlib import Path

from sklearn.utils import gen_batches
from sklearn.model_selection import train_test_split

with open(Path("refLabels.npz"), "rb") as file:
    labels = npload(file)
    labels = labels.astype(float32)
with open(Path("refTotdata.npz"), "rb") as file:
    points = npload(file)
    points = points.astype(float32)


accuracy = BinaryAccuracy()
conf_matrix = BinaryConfusionMatrix()
f1 = BinaryF1Score()
recall = BinaryRecall()
precision = BinaryPrecision()

x_train, x_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, random_state=42, shuffle=True)

batch_size = 10

batches = gen_batches(x_train.shape[0], batch_size)

x_train_batches = []
y_train_batches = []
# Create data loaders.
for gen in batches:
    x_train_batches.append(x_train[gen])
    y_train_batches.append(y_train[gen])


batches = gen_batches(x_test.shape[0], batch_size)

x_test_batches = []
y_test_batches = []
# Create data loaders.
for gen in batches:
    x_test_batches.append(x_test[gen])
    y_test_batches.append(y_test[gen])

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*250, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(x_data, y_data, model, loss_fn, optimizer):
    size = 0
    for x in x_data:
        size += x.shape[0]
    model.train()
    for batch, (X, y) in enumerate(zip(x_data, y_data)):
        X, y = from_numpy(X).to(device), from_numpy(y).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(x_data, y_data, model, loss_fn):
    accuracy.reset()
    conf_matrix.reset()
    f1.reset()
    size = 0
    for x in x_data:
        size += x.shape[0]
    num_batches = len(x_data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in zip(x_data, y_data):
            X, y = from_numpy(X).to(device), from_numpy(y).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            accuracy.update(pred.argmax(1), y.argmax(1))
            conf_matrix.update(pred.argmax(1), y.argmax(1))
            f1.update(pred.argmax(1), y.argmax(1))
            precision.update(pred.argmax(1), y.argmax(1))
            recall.update(pred.argmax(1), y.argmax(1))
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, conf_matrix, f1, recall, precision


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(x_train_batches, y_train_batches, model, loss_fn, optimizer)
    acc, conf, f1, recall, precision = test(x_test_batches, y_test_batches, model, loss_fn)
print(f" accuracy: {acc.compute()}, precision: {precision.compute()}, recall: {recall.compute()} f1: {f1.compute()}")
print("Done!")