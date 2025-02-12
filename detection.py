import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# Load the trained model
class HelmetDetectionModel(nn.Module):
    def __init__(self):
        super(HelmetDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  
        self.fc2 = nn.Linear(256, 1)  
        self.dropout = nn.Dropout(0.5)  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HelmetDetectionModel().to(device)
model.load_state_dict(torch.load("helmet_detection_model.pth", map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# OpenCV Video Capture (Live Webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)  

    # Get prediction
    with torch.no_grad():
        output = model(img)
        prediction = "Helmet" if output.item() > 0.5 else "No Helmet"

    # Display the result
    label_color = (0, 255, 0) if prediction == "Helmet" else (0, 0, 255)
    cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
    cv2.imshow("Helmet Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
