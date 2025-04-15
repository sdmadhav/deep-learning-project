import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import copy
from torchvision import datasets

# === SETUP ===
# Set directories for train, validation, and test data
train_data_dir = '/kaggle/working/spectrograms/train'
val_data_dir = '/kaggle/working/spectrograms/valid'
test_data_dir = '/kaggle/working/spectrograms/test'

# === Data transforms ===
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(val_data_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=32)

test_dataset = datasets.ImageFolder(test_data_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=32)

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SpectrogramCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectrogramCNN(num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_f1 = 0
patience = 5
patience_counter = 0
best_model = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(70):
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='weighted')

    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f"Epoch {epoch+1}: "
          f"Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
          f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "best_spectrogram_cnn.pth")
        patience_counter = 0

    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered!")
            break




def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n")
    print(report)


# After training
evaluate_model(model, test_loader, class_names=val_dataset.classes)


# !pip install torchcam

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import numpy as np

# Load your model
model.load_state_dict(torch.load("best_spectrogram_cnn.pth"))
model.eval()
from torch.nn import Conv2d

# Automatically find last Conv2d layer
target_layer = None
for name, module in reversed(list(model.named_modules())):
    if isinstance(module, Conv2d):
        target_layer = name
        break

print(f"Using layer: {target_layer}")

# Choose target layer for Grad-CAM (e.g., last conv layer)
cam_extractor = GradCAM(model, target_layer=target_layer)

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# Load and preprocess image
image_path = "/kaggle/working/spectrogramss_test/angry/01-01-05-01-01-01-14_audio.png"
img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# Forward pass and CAM generation
out = model(input_tensor)
class_id = out.argmax().item()
activation_map = cam_extractor(class_id, out)

# Overlay CAM on original image
heatmap = activation_map[0].squeeze().cpu().numpy()
result = overlay_mask(img, Image.fromarray(np.uint8(heatmap * 255), mode='L'), alpha=0.5)

# Show
plt.figure(figsize=(8, 8))
plt.imshow(result)
plt.title(f"Grad-CAM for class {class_id}")
plt.axis("off")
plt.show()