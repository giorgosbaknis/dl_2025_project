import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimpleANN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleANN, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class DeepANN(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(DeepANN, self).__init__()
        layers = []
        prev_dim = input_dim

        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            prev_dim = size

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MedicalDataset(Dataset):
    def __init__(self, df, scaler=None):
        self.labels = df['Diagnosis'].values.astype('float32')
        df = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])

        categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel']
        df = pd.get_dummies(df, columns=categorical_cols)

        binary_cols = [
            'Smoking', 'AlcoholConsumption', 'FamilyHistoryAlzheimers',
            'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury',
            'Hypertension', 'MemoryComplaints', 'BehavioralProblems',
            'Confusion', 'Disorientation', 'PersonalityChanges',
            'DifficultyCompletingTasks', 'Forgetfulness', 'ADL', 'FunctionalAssessment'
        ]
        feature_cols = df.columns.difference(binary_cols)

        if scaler is None:
            self.scaler = MinMaxScaler()
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            self.scaler = scaler
            df[feature_cols] = self.scaler.transform(df[feature_cols])

        self.features = df.values.astype('float32')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

# Load data
data = pd.read_csv('alzheimers_disease_data.csv')
data.fillna(0, inplace=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = data.drop(columns=['PatientID', 'DoctorInCharge'])
y = data['Diagnosis'].values.astype(int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

hidden_dim = 78
learning_rate = 0.1
momentum = 0.6
epochs = 100
early_stopping_patience = 10  # Εποχές χωρίς βελτίωση
r = 0.0001

# Initialize model once and save its initial state
input_dim = data.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
input_dim = pd.get_dummies(input_dim, columns=['Gender', 'Ethnicity', 'EducationLevel']).shape[1]
#input dim: 39

model = DeepANN(input_dim, [78, 39, 20]).to(device)


criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters())

train_losses_all = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🌀 Fold {fold + 1}")

    train_df = data.iloc[train_idx]
    val_df = data.iloc[val_idx]

    train_dataset = MedicalDataset(train_df)
    val_dataset = MedicalDataset(val_df, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # input_dim = train_dataset.features.shape[1]
    # print(f"Number of inputs (features) in the neural network: {input_dim}")
    # model = SimpleANN(input_dim, hidden_dim).to(device)

    # criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100
        train_losses.append(avg_train_loss)

        # Validation Loss for Early Stopping
        model.eval()
        val_loss_bce = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_bce = criterion(outputs.squeeze(), labels)
                val_loss_bce += loss_bce.item()

        avg_val_loss = val_loss_bce / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

        # # Early Stopping Check
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     best_model_state = model.state_dict()
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= early_stopping_patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    

    #NOTE - Append the train losses for the graph
    train_losses_all.append(train_losses)

    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)

    # # Plot convergence graph
    # plt.figure()
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    # plt.title(f'📈 Convergence Curve - Fold {fold + 1}')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'convergence_fold_{fold + 1}.png')
    # plt.show()

    # Final Evaluation
    model.eval()
    val_loss_bce = 0.0
    val_loss_mse = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss_bce = criterion(outputs.squeeze(), labels)
            val_loss_bce += loss_bce.item()

            loss_mse = nn.MSELoss()(outputs.squeeze(), labels)
            val_loss_mse += loss_mse.item()

            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss_bce = val_loss_bce / len(val_loader)
    avg_val_loss_mse = val_loss_mse / len(val_loader)
    val_accuracy = correct / total * 100

    print(f"Final Validation BCE Loss: {avg_val_loss_bce:.4f}, MSE Loss: {avg_val_loss_mse:.4f}, Accuracy: {val_accuracy:.2f}%")


plt.figure(figsize=(10, 6))
for fold_idx, train_losses in enumerate(train_losses_all):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label=f'Fold {fold_idx + 1}')

plt.title('Training Losses Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(title='Folds')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_losses_all_folds.png')
plt.show()

