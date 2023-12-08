from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable

import os



# BetterNet model definition
class BetterNet(torch.nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super(BetterNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Dropout2d(dropout),

            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(dropout),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(dropout),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout2d(dropout),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128*7*7, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),

            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),

            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
  
app = Flask(__name__)

device = torch.device("cpu")
model = BetterNet()
checkpoint = torch.load('checkpoint_99.73.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ... (Keep the preprocessing and model code intact)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']

    # Preprocess image data
    image_array = np.asarray(data, dtype=np.uint8).reshape(280, 280
, 4)
    image_array = image_array[:, :, 3]  # only the alpha channel
    # Resizing and transforming the image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28))
    ])
    
    resized_image = transform(image_array)

    # Displaying image BEFORE transformation and AFTER resizing
    np.set_printoptions(threshold=np.inf)
    print(np.array(resized_image))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (8.6270,)),
    ])

    image_tensor = transform(image_array)
    image_tensor = image_tensor.view(1, 1, 28, 28).float()
    image_tensor = Variable(image_tensor)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predictions = {f"{i}": float(probs[i]) for i in range(10)}

        # Getting the index (i.e., the digit) of the highest probability
        highest_prediction_index = predicted.item()


    # Print tensor array & activation array in console
    print(image_tensor)
    print(outputs)

    response = {
    'highest_prediction': highest_prediction_index,
    'predictions': predictions
}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)