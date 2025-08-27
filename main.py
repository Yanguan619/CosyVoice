import torch
import torch.nn as nn
from torch.export import export


class FullOp(nn.Module):
    def __init__(self):
        super(FullOp, self).__init__()

    def forward(self, x, index:int):
        return torch.full((3, 3), index)


def main() -> None:
    print("Hello from cosyvoice!")


# Example usage
if __name__ == "__main__":
    model = FullOp()
    example_args = (torch.tensor(5.0, dtype=torch.float32, requires_grad=True), 5)
    # print(example_args[0].requires_grad)
    exported_program = export(model, example_args)
