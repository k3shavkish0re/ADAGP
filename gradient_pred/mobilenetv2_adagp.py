import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import numpy as np
import random
import time
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from functools import partial

# Set the seed for PyTorch
torch.manual_seed(42)

# If you are using CUDA, set the seed for CUDA as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Set the seed for NumPy
np.random.seed(42)

# Set the seed for Python's random module
random.seed(42)

# Optionally, you can also set the environment variable for Python hash seeding
os.environ["PYTHONHASHSEED"] = str(42)

# Define Tensor Reorganization
def tensor_reorganization(tensor):
    if tensor.dim() == 4:  
        return tensor.permute(1, 0, 2, 3).mean(dim=1, keepdim=True)  
    elif tensor.dim() == 3:  
        return tensor.mean(dim=0, keepdim=True).unsqueeze(0)  
    elif tensor.dim() == 2:  
        return tensor.unsqueeze(0).unsqueeze(0)  
    elif tensor.dim() == 1:  
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}.")

# Define Predictor Model
class PredictorModel(nn.Module):
    def __init__(self, hidden_channels, pool_output_size, fixed_output_size):
        super(PredictorModel, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(pool_output_size)  
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)

        flattened_size = hidden_channels * pool_output_size[0] * pool_output_size[1]
        self.fc = nn.Linear(flattened_size, fixed_output_size)

    def forward(self, x, output_size):
        x = self.pool(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        pred_chunk = self.fc(x)
        pred_chunk = torch.mean(pred_chunk, dim=0, keepdim=True)
        return pred_chunk

def predict_and_fill(predictor, activations, output_size, fixed_output_size=1000):
    # Generate a predicted chunk
    chunk_output_size = torch.Size([fixed_output_size])
    pred_chunk = predictor(activations, chunk_output_size).flatten()
    total_elements = output_size.numel()  # Total elements in the desired output
    num_full_repeats = total_elements // fixed_output_size
    remainder = total_elements % fixed_output_size

    # Handle insufficient total elements gracefully
    if pred_chunk.numel() < fixed_output_size:
        raise ValueError("Predictor output size is smaller than fixed_output_size. Adjust fixed_output_size or the predictor model.")

    # Construct full prediction
    if num_full_repeats > 0:
        full_prediction = torch.cat([pred_chunk] * num_full_repeats)
    else:
        full_prediction = pred_chunk[:0]  # Start with an empty tensor if no repeats are needed

    if remainder > 0:
        remainder_fill = pred_chunk[:remainder]
        full_prediction = torch.cat((full_prediction, remainder_fill))

    # Trim or pad the prediction to ensure correct size
    full_prediction = full_prediction[:total_elements]  # Trim excess elements
    return full_prediction.view(*output_size)


import time
from functools import partial

def activation_hook(module, input, output, activation_dict, layer_name):
    """
    Hook function to capture and store activations after tensor reorganization.

    Args:
        module: The layer/module for which the hook is registered.
        input: Input tensor(s) to the module.
        output: Output tensor of the module.
        layer_name: Name of the layer to use as a key in the dictionary.
    """
    if isinstance(output, torch.Tensor):
        # Apply tensor reorganization and store
        activation_dict[layer_name] = tensor_reorganization(output.detach())
        # print(f"Captured activation for {layer_name}: {activation_dict[layer_name].shape}")


def register_hooks(model, activation_dict):
    """
    Registers hooks for capturing activations of layers with trainable biases.

    Args:
        model: The model for which hooks are to be registered.
        activation_dict: Dictionary to store activations.
    """
    for name, module in model.named_modules():
        if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
            hook_fn = partial(activation_hook, activation_dict=activation_dict, layer_name=name)
            module.register_forward_hook(hook_fn)

loss_dictionary = {} # for saving loss values per layer
ratio_sum = 4
# Training Loop with Activation and Gradient Flushing
def train_adagp(model, predictor, dataloader, num_epochs=20, warmup_epochs=5, lr=0.001, fixed_output_size=1000, save_dir="./checkpoints", device="cuda", start_epoch=0):
    model.to(device)
    predictor.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_model = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=lr * 0.1)
    scheduler_model = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=3, verbose=True)
    scheduler_predictor = ReduceLROnPlateau(optimizer_predictor, mode='min', factor=0.5, patience=3, verbose=True)

    activation_dict = {}
    register_hooks(model, activation_dict)
    
    for epoch in range(start_epoch, num_epochs):
        phase = "Warm-Up" if epoch < warmup_epochs else "Phase-GP"
        
        if epoch == warmup_epochs:
          counter = 0

        if epoch > warmup_epochs:
          counter += 1
          if counter == ratio_sum:
            counter = 0
            phase = "Warm-Up"
      

        if phase == "Warm-Up":
            predictor.train()
        else:
            predictor.eval()
        
        epoch_loss = 0.0
        predictor_epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            layer_outputs = inputs  # Initialize layer_outputs before use

            if phase == "Warm-Up":
                optimizer_model.zero_grad()
                optimizer_predictor.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                loss.backward()

                optimizer_model.step()

                # Process activations and apply predictor gradient updates
                for name, param in model.named_parameters():
                    # Extract layer name for matching
                    if '.' in name:
                        layer_name = '.'.join(name.split('.')[:-1])  # Remove the last part (e.g., 'weight')
                    else:
                        layer_name = name

                    if layer_name in activation_dict:
                        activations = activation_dict[layer_name]
                        # print(f"Using activations from {layer_name} for parameter {name}")
                        
                        # Generate predicted gradients
                        pred_grads = predict_and_fill(
                            predictor=predictor,
                            activations=activations,
                            output_size=param.grad.size(),
                            fixed_output_size=fixed_output_size
                        )
                        # print(f"Predicted gradients: {pred_grads.shape}")
                        
                        # Calculate predictor loss
                        predictor_loss = nn.MSELoss()(pred_grads, param.grad.detach())
                        predictor_loss.backward()
                        predictor_epoch_loss += predictor_loss.item()
                        
                        if name not in loss_dictionary:
                          loss_dictionary[name] = [0 for _ in range(start_epoch, num_epochs)]
                        else:
                          loss_dictionary[name][epoch] += predictor_loss.item()

                optimizer_predictor.step()

            elif phase == "Phase-GP":
                optimizer_model.zero_grad()

                # Forward pass through each layer
                for name, module in model.named_children():
                    if isinstance(module, nn.Linear) or 'classifier' in name:
                        layer_outputs = layer_outputs.view(layer_outputs.size(0), -1)  # Flatten for Linear layers
                    layer_outputs = module(layer_outputs)

                    # Handle activations and gradients
                    if hasattr(module, "bias") and module.bias.requires_grad:
                        activations = activation_dict.get(name, None)  # Use .get() to safely access activations
                        if activations is not None:
                            pred_grads = predict_and_fill(predictor=predictor, activations=activations, output_size=module.bias.size(), fixed_output_size=fixed_output_size)
                            module.bias.grad = pred_grads
                
                # Calculate loss and backpropagate
                loss = criterion(layer_outputs, labels)
                loss.backward()
                optimizer_model.step()
                epoch_loss += loss.item()

            # Print stats at regular intervals
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}, Predictor Loss: {predictor_epoch_loss / (batch_idx + 1):.4f}")


        # End of epoch
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Phase: {phase} | Model Loss: {epoch_loss:.4f} | Predictor Loss: {predictor_epoch_loss:.4f} | Time: {elapsed_time:.2f}s")

        # Step the schedulers
        scheduler_model.step(epoch_loss)
        if phase == "Warm-Up":
            scheduler_predictor.step(predictor_loss)


# Example Usage
mobilenet_v2 = models.mobilenet_v2(pretrained=False)
mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.classifier[1].in_features, 10)
predictor = PredictorModel(pool_output_size=(4, 4), hidden_channels=64, fixed_output_size=10000)

# CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Train the ADA-GP model
train_adagp(mobilenet_v2, predictor, trainloader, num_epochs=20, warmup_epochs=5, lr=0.001, fixed_output_size=10000, device='cuda' if torch.cuda.is_available() else 'cpu')

import json
with open("loss_dict.json", "w") as f:
    json.dump(loss_dictionary, f)

# Define Testing Logic
def test_model(model, dataloader, device="cuda"):
    """
    Evaluate the model on the test dataset.

    Args:
        model: Trained model to be evaluated.
        dataloader: DataLoader for the test dataset.
        device: Device to perform the evaluation on.

    Returns:
        test_loss: Average loss on the test dataset.
        test_accuracy: Accuracy on the test dataset.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # No need to calculate gradients during testing
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Calculate predictions and accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_loss /= len(dataloader)  # Average loss
    test_accuracy = 100.0 * correct_predictions / total_samples

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
    return test_loss, test_accuracy


# CIFAR-10 Test Dataset
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Evaluate the trained model
test_model(mobilenet_v2, testloader, device='cuda' if torch.cuda.is_available() else 'cpu')