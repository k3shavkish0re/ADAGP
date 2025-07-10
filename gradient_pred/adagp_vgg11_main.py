import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np
import random
from torchvision.models import vgg11, VGG11_Weights

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
import os
os.environ["PYTHONHASHSEED"] = str(42)


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
      return tensor.permute(1, 0, 2, 3).mean(dim=1, keepdim=True)  # Result: (1, channels, height, width)

    elif tensor.dim() == 3:  # For FC layer weights with batch dimension (3D tensors)
        return tensor.mean(dim=0, keepdim=True).unsqueeze(0)  # Result: (1, 1, output_features, input_features)

    elif tensor.dim() == 2:  # For FC layer weights without batch dimension
        return tensor.unsqueeze(0).unsqueeze(0)  # Result: (1, 1, output_features, input_features)

    elif tensor.dim() == 1:  # For FC layer biases
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Result: (1, 1, 1, output_features)

    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Only 2D, 3D, and 4D tensors are supported.")


# Define Predictor Model
class PredictorModel(nn.Module):
    def __init__(self, hidden_channels, pool_output_size, fixed_output_size):
        """
        Predictor Model for ADA-GP with fixed output size.

        Args:
            hidden_channels (int): Number of hidden channels for the convolutional layers.
            pool_output_size (tuple): Output size of the AdaptiveMaxPool2d layer (e.g., (4, 4)).
            fixed_output_size (int): Fixed size of the predicted gradient.
        """
        super(PredictorModel, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(pool_output_size)  # Dynamically reduce spatial dimensions
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)

        # Dynamically calculate the flattened size after pool and conv2
        flattened_size = hidden_channels * pool_output_size[0] * pool_output_size[1]
        self.fc = nn.Linear(flattened_size, fixed_output_size)

    def forward(self, x, output_size):
        """
        Forward pass of the predictor.

        Args:
            x (torch.Tensor): Input tensor (from activations or reorganized gradients).
            output_size (torch.Size): Target size of the gradient tensor.

        Returns:
            torch.Tensor: Predicted gradients reshaped to match `output_size`.
        """
        x = self.pool(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        # print(x.shape)
        pred_chunk = self.fc(x)
        pred_chunk = torch.mean(pred_chunk, dim=0, keepdim=True)
        # print(pred_chunk.shape)
        return pred_chunk




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
    # Predict a fixed-size chunk
    chunk_output_size = torch.Size([fixed_output_size])
    # print(activations.shape)
    pred_chunk = predictor(activations, chunk_output_size).flatten()  # Ensure 1D output
    # print(f"pred chuck : {pred_chunk.shape}")

    total_elements = output_size.numel()
    # print(type(total_elements))
    # print(total_elements)
    num_full_repeats = total_elements // fixed_output_size
    # print(f"Full repeats : {num_full_repeats}")
    remainder = total_elements % fixed_output_size
    # print(remainder)
    #handles the case if our target tensor is smaller than the number of tensors we need
    if num_full_repeats == 0:
      full_prediction = pred_chunk[:total_elements]
      return full_prediction.view(*output_size)
    # print(remainder)

    # Repeat full chunks using concatenation
    repeated_prediction = torch.cat([pred_chunk] * num_full_repeats)
    # print(repeated_prediction.shape)

    # Handle remainder
    if remainder > 0:
        remainder_fill = pred_chunk[:remainder]  # Slice the required number of elements for the remainder
        full_prediction = torch.cat((repeated_prediction, remainder_fill))
        # print(full_prediction.shape)
    else:
        full_prediction = repeated_prediction

    # Reshape to match the target gradient size
    # print("full pred",full_prediction.shape)
    return full_prediction.view(*output_size)


# Training Loop with Activation and Gradient Flushing
import time


def activation_hook(module, input, output, activation_dict, layer_name):
    """
    Hook function to capture and store activations after tensor reorganization.

    Args:
        module: The layer/module for which the hook is registered.
        input: Input tensor(s) to the module.
        output: Output tensor of the module.
        layer_name: Name of the layer to use as a key in the dictionary.
    """
    # global activation_dict
    if isinstance(output, torch.Tensor):
        # Apply tensor reorganization and store
        activation_dict[layer_name] = tensor_reorganization(output.detach())

from functools import partial

def register_hooks(model, activation_dict):
    """
    Registers hooks for capturing activations of layers with trainable weights.

    Args:
        model: The model for which hooks are to be registered.
        activation_dict: Dictionary to store activations.
    """
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.requires_grad:
            # Use partial to bind the layer_name to the hook
            hook_fn = partial(activation_hook, activation_dict=activation_dict, layer_name=name)
            module.register_forward_hook(hook_fn)



def save_checkpoint(
    model, predictor, optimizer_model, optimizer_predictor, scheduler_model, scheduler_predictor, epoch, phase, save_path
):
    checkpoint = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_model_state_dict": optimizer_model.state_dict(),
        "optimizer_predictor_state_dict": optimizer_predictor.state_dict(),
        "scheduler_model_state_dict": scheduler_model.state_dict(),
        "scheduler_predictor_state_dict": scheduler_predictor.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    model, predictor, optimizer_model, optimizer_predictor, scheduler_model, scheduler_predictor, load_path, device="cuda"
):

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    optimizer_model.load_state_dict(checkpoint["optimizer_model_state_dict"])
    optimizer_predictor.load_state_dict(checkpoint["optimizer_predictor_state_dict"])
    scheduler_model.load_state_dict(checkpoint["scheduler_model_state_dict"])
    scheduler_predictor.load_state_dict(checkpoint["scheduler_predictor_state_dict"])
    print(f"Checkpoint loaded: {load_path}")
    return checkpoint["epoch"], checkpoint["phase"]

from torch.optim.lr_scheduler import ReduceLROnPlateau

loss_dictionary = {} # for saving loss values per layer

use_checkpoint = False
def train_adagp(model, predictor, dataloader, num_epochs=20, warmup_epochs=5, lr=0.001, fixed_output_size=1000, save_dir="./checkpoints", device="cuda", start_epoch=0):
    model.to(device)
    predictor.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_model = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=lr * 0.1)

    scheduler_model = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=3, verbose=True)
    scheduler_predictor = ReduceLROnPlateau(optimizer_predictor, mode='min', factor=0.5, patience=3, verbose=True)


    if use_checkpoint:
        # Load the checkpoint
        load_path = "/content/checkpoints/checkpoint_epoch_1_Warm-Up.pt"
        last_epoch, last_phase = load_checkpoint(
            model, predictor, optimizer_model, optimizer_predictor, scheduler_model, scheduler_predictor, load_path
        )

        print(f"Resumed training from epoch {last_epoch + 1}, phase: {last_phase}")

        # Update training parameters if necessary
        start_epoch = last_epoch + 1  # Resume from the next epoch
        num_epochs = 10  # Total number of epochs

    activation_dict = {}
    ratio_sum = 4
    register_hooks(model, activation_dict)
    # counter = 0
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

            # Warm-Up Phase
            if phase == "Warm-Up":
                optimizer_model.zero_grad()
                optimizer_predictor.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

                loss.backward()
                optimizer_model.step()

                # print(activation_dict.keys())

                 # Train predictor
                for name, param in model.named_parameters():
                  # print(name)
                  param_key = '.'.join(name.split('.')[:-1])
                  # is_conv = 'conv' in name

                  if param.grad is not None and param_key in activation_dict:
                    # print(name)
                    activations = activation_dict[param_key]
                    # print(activations.shape)
                    if len(activations.shape) == 1:
                      continue
                    pred_grads = predict_and_fill(
                        predictor=predictor,
                        activations=activations,
                        output_size=param.grad.size(),
                        fixed_output_size=fixed_output_size
                    )
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
                layer_outputs = inputs

                # Perform a layer-by-layer forward pass
                for name, module in model.named_children():
                    # print(f"Layer name :{name}, layer_outputs.shape: {layer_outputs.shape}")
                    if isinstance(module, nn.Linear) or 'classifier' in name:
                      layer_outputs = layer_outputs.view(layer_outputs.size(0), -1)  # Flatten to (batch_size, features)

                    # print(f"Layer name :{name}, layer_outputs.shape: {layer_outputs.shape}")

                    layer_outputs = module(layer_outputs)  # Forward pass for the current layer
                    # print(layer_outputs.shape)


                    # Check if module has weights and requires gradient
                    if hasattr(module, "weight") and module.weight.requires_grad:
                        activations = tensor_reorganization(layer_outputs)
                        # print(activations.shape)
                        pred_grads = predict_and_fill(
                            predictor=predictor,
                            activations=activations,
                            output_size=module.weight.size(),
                            fixed_output_size=fixed_output_size
                        )
                        module.weight.grad = pred_grads  # Assign predicted gradients

                # Compute loss after full forward pass
                loss = criterion(layer_outputs, labels)
                loss.backward()
                optimizer_model.step()
                epoch_loss += loss.item()

            # Print batch-level stats
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Predictor Loss: {predictor_epoch_loss / (batch_idx + 1):.4f}")

            # Track accuracy
            if phase == "Warm-Up":
                _, predicted = torch.max(outputs, 1)
            else:
                # print(layer_outputs.shape)
                _, predicted = torch.max(layer_outputs, 1)
            
            # print(predicted.shape)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Epoch-level stats
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Phase: {phase} | Model Loss: {epoch_loss:.4f} | "
              f"Predictor Loss: {predictor_epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}% | Time: {elapsed_time:.2f}s")

        scheduler_model.step(epoch_loss)
        if phase == "Warm-Up":
            scheduler_predictor.step(predictor_loss)

        save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}_{phase}.pt")
        save_checkpoint(
            model,
            predictor,
            optimizer_model,
            optimizer_predictor,
            scheduler_model,
            scheduler_predictor,
            epoch,
            phase,
            save_path
        )

        print(f"Model(s), Optimizer(s), Scheduler(s) state dict saved successfully at {save_path}")
        # return loss_dictionary



# Example Usage
# resnet = resnet18(pretrained=False)
# resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# # Remove the max pooling layer to preserve resolution for small images
# resnet.maxpool = nn.Identity()

# # Modify the fully connected layer for 10 classes
# resnet.fc = nn.Linear(resnet.fc.in_features, 10)
# Load VGG11
vgg = vgg11(pretrained=False)
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 10)
predictor = PredictorModel(pool_output_size=(4, 4), hidden_channels=64, fixed_output_size=10000)

# CIFAR-10 Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for CIFAR-10
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


# Train the ADA-GP model
train_adagp(vgg, predictor, trainloader, num_epochs=20, warmup_epochs=5, device="cuda" if torch.cuda.is_available() else "cpu", fixed_output_size=10000)


import json
with open("loss_dict.json", "w") as f:
    json.dump(loss_dictionary, f)



