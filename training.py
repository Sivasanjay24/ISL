import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Constants ---
DATA_FILE = 'hand_sign_data.csv'
MODEL_PATH = 'landmark_model.pth'
LABEL_MAP_PATH = 'label_map.pkl'
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


df = pd.read_csv(DATA_FILE, header=None)
X = df.iloc[:, 1:].values.astype(np.float32)  
y_raw = df.iloc[:, 0].values                   
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(np.unique(y))

label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
with open(LABEL_MAP_PATH, 'wb') as f:
    pickle.dump(label_map, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(LandmarkDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(LandmarkDataset(X_test, y_test), batch_size=BATCH_SIZE)


class LandmarkModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LandmarkModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


model = LandmarkModel(input_size=X.shape[1], num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print(f"Training on {DEVICE}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / len(X_test)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH} and label map to {LABEL_MAP_PATH}")