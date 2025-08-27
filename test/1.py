import time

import torch
from torch import Tensor


@torch.compile
def bar(x: Tensor, y: Tensor):
    z = torch.randn(x.size(0), y.size(1))
    return torch.sin(x) * torch.cos(y) + z


t = torch.randn(10, 100)


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x: Tensor):
        return torch.nn.functional.relu(self.lin(x))


start_time = time.time()
for i in range(10):
    mod = MyModule()
    mod = torch.compile(mod, dynamic=True, fullgraph=True)
    # output = bar(torch.randn(10, 10), torch.randn(10, 10))
    print(f"{time.time() - start_time:0f}")
    start_time = time.time()


# torch._dynamo.config.capture_scalar_outputs = True


# @torch.compile(backend="eager", fullgraph=False)
# def f(xs):
#     x1, x2 = xs
#     res = torch.arange(x1, x2)
#     return res


xs = torch.tensor([4])
# res = f(xs)

a = torch.randn([4], requires_grad=True)
b = torch.randn([4], requires_grad=True)
c = torch.cat([b, a], dim=0)
print(c.requires_grad)
