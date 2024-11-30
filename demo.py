import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--cloud", type=str, default="False", nargs="?", help="Whether to run the script on cloud or not")
args = argparser.parse_args()
if args.cloud:
    args.cloud = False
    if args.cloud == 'True':
        args.cloud = True

import pandas as pd
import requests
if not args.cloud:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.net import Net


def predict(model_path, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    data = torch.tensor(data.values, dtype=torch.long).to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in loader:
            x = x[0].to(device)
            y_pred = model(x)
            y_pred = (y_pred >= 0.5).float()
            predictions.append(y_pred.cpu().numpy())
    return predictions

def send_to_aws(data):
    url = 'http://<your-ec2-instance-public-ip>:5000/predict'
    response = requests.post(url, json=data)
    return response.json()

def main():
    # Load a subset of the validation dataset
    validation_data = pd.read_csv('data/demo_features.csv')

    # Convert data to JSON format
    data_json = validation_data.to_json(orient='split')

    # Send data to AWS EC2 for inference
    if args.cloud:
        predictions = send_to_aws(data_json)[0]
    else:
        predictions = predict('model/model_lr_1e-2.pth', validation_data)[0]

    y = pd.read_csv('data/demo_labels.csv').values
    print("Sample\tPrediction\tActual\tLabel")
    true = 0
    for i in range(len(y)):
        print(f"Sample {i+1}:\t{int(predictions[i][0])},\tActual:\t{y[i][0]}")
        if y[i] == predictions[i]:
            true += 1
    print(f'Accuracy: {true/len(y)}')
    print(f"TP: {sum([1 for i in range(len(y)) if y[i][0] == 1 and predictions[i][0] == 1])}")
    print(f"FP: {sum([1 for i in range(len(y)) if y[i][0] == 0 and predictions[i][0] == 1])}")
    print(f"TN: {sum([1 for i in range(len(y)) if y[i][0] == 0 and predictions[i][0] == 0])}")
    print(f"FN: {sum([1 for i in range(len(y)) if y[i][0] == 1 and predictions[i][0] == 0])}")
    print(f"Precision: {sum([1 for i in range(len(y)) if y[i][0] == 1 and predictions[i][0] == 1]) / sum([1 for i in range(len(y)) if predictions[i][0] == 1])}")
    print(f"Recall: {sum([1 for i in range(len(y)) if y[i][0] == 1 and predictions[i][0] == 1]) / sum([1 for i in range(len(y)) if y[i][0] == 1])}")

if __name__ == '__main__':
    main()
