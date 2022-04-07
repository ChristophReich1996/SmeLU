import torch
import torch.nn as nn

from smelu import SmeLU


def relu_runtime() -> None:
    network = nn.Sequential(*[nn.ReLU() for _ in range(100)])
    network.cuda()
    input = torch.randn(1, 3, 1024, 1024, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = network(input)
    output.sum().backward()
    end.record()
    torch.cuda.synchronize()
    print("ReLU (100x) runtime (forward and backward)", start.elapsed_time(end))


def smelu_runtime() -> None:
    network = nn.Sequential(*[SmeLU() for _ in range(100)])
    network.cuda()
    input = torch.randn(1, 3, 1024, 1024, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = network(input)
    output.sum().backward()
    end.record()
    torch.cuda.synchronize()
    print("SmeLU (100x) runtime (forward and backward)", start.elapsed_time(end))


def main() -> None:
    relu_runtime()
    smelu_runtime()


if __name__ == '__main__':
    main()
