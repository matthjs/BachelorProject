# Check if pytorch works as expected
import torch
import torch.nn as nn


def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Define a simple neural network and move it to the GPU
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1)).to(device)

    # Example of using the model
    input_data = torch.randn(1, 10).to(device)
    output = model(input_data)
    print("Output:", output.item())


if __name__ == "__main__":
    main()
