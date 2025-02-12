import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os

# Define data paths
DATASET_PATH =r"C:\Users\rahul\OneDrive\Desktop\Helmet_project\Helmet_detection\ai_helmet_detection\helmet_detection_dataset"

# Define image transformations for augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load dataset
train_data = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "train"), transform=transform)
val_data = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "val"), transform=transform)
test_data = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "test"), transform=transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define a CNN Model for Helmet Detection
class HelmetDetectionModel(nn.Module):
    def __init__(self):
        super(HelmetDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjusted based on 128x128 input
        self.fc2 = nn.Linear(256, 1)  # Binary classification (Helmet or No Helmet)
        self.dropout = nn.Dropout(0.5)  # Prevent overfitting

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HelmetDetectionModel().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "helmet_detection_model.pth")
print("Model saved successfully!")

# Evaluate on test data
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.view(-1, 1)).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
