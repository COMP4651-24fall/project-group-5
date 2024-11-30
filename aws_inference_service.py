from flask import Flask, request, jsonify
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.net import Net

app = Flask(__name__)

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    return model

model = load_model('model/model_lr_1e-2.pth')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data_df = pd.read_json(data, orient='split')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data_df.values, dtype=torch.long).to(device)
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in loader:
            x = x[0].to(device)
            y_pred = model(x)
            y_pred = (y_pred >= 0.5).float()
            predictions.append(y_pred.cpu().numpy().tolist())
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)