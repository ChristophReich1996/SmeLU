import torch
import matplotlib.pyplot as plt

from smelu import SmeLU


def main() -> None:
    # Init figures
    fig, ax = plt.subplots(1, 1)
    fig_grad, ax_grad = plt.subplots(1, 1)
    # Iterate over some beta values
    for beta in [0.5, 1., 2., 3., 4.]:
        # Init SemLU
        smelu: SmeLU = SmeLU(beta=beta)
        # Make input
        input: torch.Tensor = torch.linspace(-6, 6, 1000, requires_grad=True)
        # Get activations
        output: torch.Tensor = smelu(input)
        # Compute gradients
        output.sum().backward()
        # Plot activation and gradients
        ax.plot(input.detach(), output.detach(), label=str(beta))
        ax_grad.plot(input.detach(), input.grad.detach(), label=str(beta))
    # Show legend, title and grid
    ax.legend()
    ax_grad.legend()
    ax.set_title("SemLU")
    ax_grad.set_title("SemLU gradient")
    ax.grid()
    ax_grad.grid()
    # Show plots
    plt.show()


if __name__ == '__main__':
    main()
