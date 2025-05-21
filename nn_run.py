import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Neural Network Architecture (Enhanced)
class GravNet(nn.Module):
    def __init__(self):
        super(GravNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(11, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.GELU(),
            
            nn.Linear(128, 4)
        )
        
        # Initialize weights - FIXED INITIALIZATION
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # Changed 'gelu' to 'relu'
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.main(x)

# Dataset Handler
class PhysicsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)

# Enhanced Training System
class TrainingSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_and_preprocess(self, train_file='two_body_data.csv', test_file='test_data.csv'):
        # Load data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Clean data
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        # Split features and targets
        X_train = train_df.iloc[:, :11].values
        y_train = train_df.iloc[:, -4:].values
        X_test = test_df.iloc[:, :11].values
        y_test = test_df.iloc[:, -4:].values
        
        # Fit scalers on training data
        self.scaler_x.fit(X_train)
        self.scaler_y.fit(y_train)
        
        # Transform data
        X_train = self.scaler_x.transform(X_train)
        X_test = self.scaler_x.transform(X_test)
        y_train = self.scaler_y.transform(y_train)
        y_test = self.scaler_y.transform(y_test)
        
        # Split validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        return (X_train, y_train, X_val, y_val, X_test, y_test)

    def train_model(self, X_train, y_train, X_val, y_val):
        model = GravNet().to(self.device)
        criterion = nn.HuberLoss(delta=0.5)  # More robust than MSE
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        train_dataset = PhysicsDataset(X_train, y_train)
        val_dataset = PhysicsDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=512, pin_memory=True)
        
        best_loss = float('inf')
        early_stop_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(200):
            model.train()
            train_loss = 0
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping and model checkpoint
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= 10:
                    print("Early stopping triggered")
                    break
            
            scheduler.step(avg_val_loss)
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
         # Inside train_model() after training completes:
        self.save_scalers()
        return model

    def evaluate_model(self, model, X_test, y_test):
        model.eval()
        test_dataset = PhysicsDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=512)
        
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs).cpu().numpy()
                predictions.append(outputs)
                true_values.append(targets.numpy())
        
        y_pred = np.vstack(predictions)
        y_true = np.vstack(true_values)
        
        # Inverse transform for original scale
        y_pred = self.scaler_y.inverse_transform(y_pred)
        y_true = self.scaler_y.inverse_transform(y_true)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print("\nTest Results:")
        print(f"MAE: {mae:.2f} m")
        print(f"RMSE: {rmse:.2f} m")
        print(f"R² Score: {r2:.4f}")
        
        # Save predictions
        results_df = pd.DataFrame({
            'True_x1': y_true[:, 0],
            'Pred_x1': y_pred[:, 0],
            'True_y1': y_true[:, 1],
            'Pred_y1': y_pred[:, 1],
            'True_x2': y_true[:, 2],
            'Pred_x2': y_pred[:, 2],
            'True_y2': y_true[:, 3],
            'Pred_y2': y_pred[:, 3]
        })
        results_df.to_csv('predictions_vs_actuals.csv', index=False)

    def save_scalers(self):
        torch.save(self.scaler_x, 'scaler_x.pth')
        torch.save(self.scaler_y, 'scaler_y.pth')
        
        return mae, rmse, r2

class PredictionSystem:
    def __init__(self, scaler_x=None, scaler_y=None, model_path='best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_x = scaler_x if scaler_x is not None else torch.load('scaler_x.pth')
        self.scaler_y = scaler_y if scaler_y is not None else torch.load('scaler_y.pth')
        self.model = GravNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    

# Modified main block
if __name__ == "__main__":
    trainer = TrainingSystem()
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_preprocess()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # Check if model exists
    model_path = 'best_model.pth'
    train_new = not os.path.exists(model_path)
    
    if train_new:
        print("No existing model found. Starting training...")
        try:
            model = trainer.train_model(X_train, y_train, X_val, y_val)
        except KeyboardInterrupt:
            print("Training interrupted")
    else:
        print("Loading existing model...")
        model = GravNet().to(trainer.device)
        model.load_state_dict(torch.load(model_path))

    # Always evaluate current model
    trainer.evaluate_model(model, X_test, y_test)

   
    
    # Prediction system
    predictor = PredictionSystem(trainer.scaler_x, trainer.scaler_y, model_path)
    
    # Rest of your prediction code...