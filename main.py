# This is a sample Python script.
import torch.cuda
from torch import nn


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(input)
    print(target)
    output = loss(input, target)
    output.backward()
    # Example of target with class probabilities
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    output = loss(input, target)
    output.backward()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
