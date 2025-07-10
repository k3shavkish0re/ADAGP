import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time




def set_seed(seed = 42):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Setting seed of {seed}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for pre-trained models
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# dataset that is to be used : CIFAR 10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# training, testing and validation set
# train_size = int(0.8 * len(trainset))
# val_size = len(trainset) - train_size
# trainset, valset = random_split(trainset, [train_size, val_size])


# dataloader for loading dataset into the network
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
# valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

#deciding the model
vgg = models.vgg11(pretrained=False)
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 10)
vgg = vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)  # Using SGD with momentum
# optimizer_model = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler_model = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


# test_loss, test_accuracy = evaluate_model_loss(model, testloader, criterion)
start_time = time.time()




def train_model(
        model, 
        trainloader, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=10, 
        starting_epoch=0, 
        device="cuda",
        seed=42):
    
    for epoch in range(starting_epoch, starting_epoch+num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        t1 = time.time()
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        t2 = time.time()

        per_epoch_time = t2-t1
        print(f"Took {per_epoch_time} seconds/epoch")

        # Compute average training loss
        avg_train_loss = running_loss / len(trainloader)

        # Compute validation loss and accuracy
        #val_loss, val_accuracy = evaluate_model_loss(model, testloader, criterion, "cuda")

        # Step the scheduler based on validation loss
        scheduler.step(avg_train_loss)


        checkpoint_path = f"./resnet_epoch_{epoch+1}.pth"
        # checkpoint_path = os.path.join(checkpoint_dir, f"resnet_epoch_{epoch+1}.pth")

        torch.save({
            'seed': seed,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save the scheduler state
            'train_loss': avg_train_loss,
            'per_epoch_time': per_epoch_time
        }, checkpoint_path)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
    return model

train_model(
    vgg,
    trainloader,
    criterion,
    optimizer,
    scheduler_model,
    20,
    0
)
end_time = time.time()
print(f"It took around {end_time - start_time} seconds for training model")
# print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


    