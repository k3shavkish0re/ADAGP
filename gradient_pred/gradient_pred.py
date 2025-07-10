import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18


# Define Tensor Reorganization
def tensor_reorganization(tensor):
    """
    Generalize tensor reorganization for both 2D, 3D, and 4D tensors.

    Args:
        tensor (torch.Tensor): Input tensor (e.g., activations, weights).
            - 4D tensor: (batch_size, channels, height, width).
            - 3D tensor: (batch_size, output_features, input_features).
            - 2D tensor: (output_features, input_features).

    Returns:
        torch.Tensor: Reorganized tensor in 4D format.
    """
    if tensor.dim() == 4:  # For convolutional weights (4D tensors)
        return tensor.mean(dim=0, keepdim=True)  # Result: (1, channels, height, width)

    elif tensor.dim() == 3:  # For FC layer weights with batch dimension (3D tensors)
        return tensor.mean(dim=0, keepdim=True).unsqueeze(0)  # Result: (1, 1, output_features, input_features)

    elif tensor.dim() == 2:  # For FC layer weights without batch dimension
        return tensor.unsqueeze(0).unsqueeze(0)  # Result: (1, 1, output_features, input_features)

    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Only 2D, 3D, and 4D tensors are supported.")


# Define Predictor Model
class PredictorModel(nn.Module):
    def __init__(self, hidden_channels, pool_output_size, fixed_output_size):
        """
        Predictor Model for ADA-GP with fixed input channels.

        Args:
            hidden_channels (int): Number of hidden channels for the convolutional layers.
            pool_output_size (tuple): Output size of the AdaptiveMaxPool2d layer (e.g., (4, 4)).
            fixed_output_size (int): Fixed size of the predicted gradient.
        """
        super(PredictorModel, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(pool_output_size)  # Dynamically reduce spatial dimensions
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1)  # Input channels fixed to 1
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        # Output size of pool_output_size after conv2 is hidden_channels * H * W (flattened)
        self.fc = nn.Linear(hidden_channels * pool_output_size[0] * pool_output_size[1], fixed_output_size)

    def forward(self, x, output_size):
        """
        Forward pass of the predictor.

        Args:
            x (torch.Tensor): Input tensor (from activations or reorganized gradients).
            output_size (torch.Size): Target size of the gradient tensor.

        Returns:
            torch.Tensor: Predicted gradients reshaped to match `output_size`.
        """
        x = self.pool(x)  # Reduce dimensions
        x = torch.relu(self.bn1(self.conv1(x)))  # First convolution
        x = torch.relu(self.bn2(self.conv2(x)))  # Second convolution
        x = torch.flatten(x, 1)  # Flatten for FC layer
        pred_chunk = self.fc(x)  # Predict fixed-size gradient chunk
        return pred_chunk



# Predict and Fill
def predict_and_fill(predictor, activations, output_size, fixed_output_size=1000):
    """
    Predict fixed-size gradients and repeat to match the target gradient size,
    filling the remaining gradients with the last prediction.

    Args:
        predictor (nn.Module): The predictor model.
        activations (torch.Tensor): Input activations for prediction.
        output_size (torch.Size): Target size of the gradient tensor.
        fixed_output_size (int): Fixed output size of the predictor.

    Returns:
        torch.Tensor: Predicted gradient tensor matching `output_size`.
    """
    chunk_output_size = torch.Size([fixed_output_size])
    pred_chunk = predictor(activations, chunk_output_size)  # Predict fixed-size gradient

    total_elements = output_size.numel()
    num_full_repeats = total_elements // fixed_output_size
    remainder = total_elements % fixed_output_size

    repeated_prediction = pred_chunk.repeat(num_full_repeats)  # Repeat predictions
    if remainder > 0:
        remainder_fill = pred_chunk[-1].repeat(remainder)  # Fill remaining elements with last prediction
        full_prediction = torch.cat((repeated_prediction, remainder_fill))
    else:
        full_prediction = repeated_prediction

    return full_prediction.view(*output_size)  # Reshape to match gradient shape


# Training Loop with Activation and Gradient Flushing
import time

def train_adagp(model, predictor, dataloader, num_epochs=20, warmup_epochs=5, lr=0.001, fixed_output_size=1000, device="cuda"):
    model.to(device)
    predictor.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_model = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=lr * 0.1)

    for epoch in range(num_epochs):
        phase = "Warm-Up" if epoch < warmup_epochs else "Gradient Prediction"
        model.train()
        predictor.train()

        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Optimizer zeroing
            optimizer_model.zero_grad()
            optimizer_predictor.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if phase == "Warm-Up":
                # Backpropagation for model gradients
                loss.backward()
                optimizer_model.step()

                # Train predictor
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        activations = tensor_reorganization(param)
                        pred_grads = predict_and_fill(
                            predictor=predictor,
                            activations=activations,
                            output_size=param.grad.size(),
                            fixed_output_size=fixed_output_size
                        )
                        predictor_loss = nn.MSELoss()(pred_grads, param.grad.detach())
                        predictor_loss.backward()
                optimizer_predictor.step()

            elif phase == "Gradient Prediction":
                # Parallel prediction of gradients with forward pass
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        activations = tensor_reorganization(param)
                        pred_grads = predict_and_fill(
                            predictor=predictor,
                            activations=activations,
                            output_size=param.size(),
                            fixed_output_size=fixed_output_size
                        )
                        param.grad = pred_grads
                loss.backward()
                optimizer_model.step()

            # Print batch-level stats
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # Clear memory
            outputs = None
            loss = None
            torch.cuda.empty_cache()

        # Epoch-level stats
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Phase: {phase} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}% | Time: {elapsed_time:.2f}s")



# Example Usage
resnet = resnet18(pretrained=False)
predictor = PredictorModel(input_channels=1, hidden_channels=128, fixed_output_size=1000)

# CIFAR-10 Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for CIFAR-10
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


# Train the ADA-GP model
train_adagp(resnet, predictor, trainloader, num_epochs=10, warmup_epochs=2, device="cuda" if torch.cuda.is_available() else "cpu")
