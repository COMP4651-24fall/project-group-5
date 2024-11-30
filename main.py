# Dependency from other file
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--cuda", type=str, default="7", nargs="?", help="CUDA devices to use, e.g., '0,1,2,3'")
args = argparser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from src.consolidate import consolidate_data
from src.transform import transform_data
from src.net import Net
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd
from tqdm import tqdm

def train_model(training_data, training_labels, validation_data, validation_labels, lr, epochs, batch_size=1024, early_stopping_rounds=10, early_stopping_threshold=0.01):
    '''
    training_data: pandas dataframe, dimensions: (n_samples, n_features), note: all columns are in integer format
    training_labels: pandas dataframe, dimensions: (n_samples, 1), note: in integer format
    validation_data: pandas dataframe, dimensions: (m_samples, n_features), note: all columns are in integer format
    validation_labels: pandas dataframe, dimensions: (m_samples, 1), note: in integer format
    lr: float, learning rate
    epochs: int, number of epochs
    batch_size: int, batch size
    early_stopping_rounds: int, number of epochs to wait before stopping training if validation loss does not improve
    early_stopping_threshold: float, minimum improvement in validation loss to be considered as improvement
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = Net().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    best_loss = float('inf')
    epochs_no_improve = 0
    
    train_data = torch.tensor(training_data.values, dtype=torch.long).to(device)
    train_labels = torch.tensor(training_labels.values, dtype=torch.float32).unsqueeze(1).to(device)
    val_data = torch.tensor(validation_data.values, dtype=torch.long).to(device)
    val_labels = torch.tensor(validation_labels.values, dtype=torch.float32).unsqueeze(1).to(device)
    training_dataset = TensorDataset(train_data, train_labels)
    validation_dataset = TensorDataset(val_data, val_labels)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    for epoch in tqdm(range(epochs)):
        model.train()
        training_loss = 0
        for x, y in tqdm(training_loader):
            x, y = x.to(device), y.to(device)  # Move tensors to the correct device
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        
        model.eval()
        validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(validation_loader):
                x, y = x.to(device), y.to(device)  # Move tensors to the correct device
                y_pred = model(x)
                loss = criterion(y_pred, y)
                validation_loss += loss.item()
                predicted = (y_pred >= 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        validation_accuracy = correct / total
        print(f'Epoch: {epoch+1}/{epochs}, Training Loss: {training_loss}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')
        
        scheduler.step(validation_loss)  # Adjust learning rate based on validation loss

        if (epoch + 1) % 10 == 0:
            print(f'Saving checkpoint at epoch {epoch+1}/{epochs}')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': training_loss
            }
            torch.save(checkpoint, f'model/training/checkpoint_{epoch+1}.pth')

        if validation_loss < best_loss - early_stopping_threshold:
            best_loss = validation_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_rounds:
                torch.save(model.state_dict(), 'model/training/early_stopping.pth')
                print(f'Early stopping at epoch {epoch+1}/{epochs} with validation loss: {validation_loss} and validation accuracy: {validation_accuracy}')
                break

def predict(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = torch.load('model/<model_name>.pth')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)
    test_data = torch.tensor(x.values, dtype=torch.long).to(device)
    test_dataset = TensorDataset(test_data)
    with torch.no_grad():
        pred = torch.tensor([]).to(device)
        for x in test_dataset:
            x = x.to(device)  # Move tensors to the correct device
            y_pred = model.forward(x)
            pred = torch.cat((pred, y_pred), 0)
    return pred


def main():
    train = pd.read_csv('data/train_transformed.csv')
    training, validation = train_test_split(train, test_size=0.2)
    training_labels, validation_labels = training["target"], validation["target"]
    training.drop("target", axis=1, inplace=True)
    validation.drop("target", axis=1, inplace=True)
    lr, epochs = 1e-2, 100
    train_model(training, training_labels, validation, validation_labels, lr, epochs, batch_size=800000)
    
if __name__ == '__main__':
    main()