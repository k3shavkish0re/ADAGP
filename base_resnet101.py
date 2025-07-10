import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import random
import numpy as np
import time
import os

# Set a seed for reproducibility
seed = 50

# Set the seed for Python's random module
random.seed(seed)

# Set the seed for NumPy
np.random.seed(seed)

# Set the seed for PyTorch
torch.manual_seed(seed)

# If using CUDA, also set the seed for all GPU operations
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load ResNet101 model without pre-trained weights
model = models.resnet101(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 (10 output classes)
model = model.to(device)

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])


# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set up ReduceLROnPlateau scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Checkpoint functions
def save_checkpoint(epoch, model, optimizer, loss, filename="checkpoint_resnet101.pth"):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch+1}.")

def load_checkpoint(filename="checkpoint_resnet101.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with loss {loss:.4f}.")
        return start_epoch, loss
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, None

# Optionally load from checkpoint
start_epoch, _ = load_checkpoint()

# Training loop parameters
num_epochs = 40

# Training loop
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    avg_loss = running_loss / len(train_loader)

    # Save checkpoint at the end of each epoch
    save_checkpoint(epoch, model, optimizer, avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time Taken: {epoch_time:.2f} seconds')

    # Validation step: Calculate test loss
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Validation Loss: {avg_test_loss:.4f}')

    # Step the scheduler based on validation loss
    scheduler.step(avg_test_loss)

# Evaluation on the test set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the CIFAR-10 test dataset: {accuracy:.2f}%')

