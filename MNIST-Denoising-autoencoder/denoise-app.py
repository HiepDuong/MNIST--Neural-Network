from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import io

app = Flask(__name__)
CORS(app)

# Load denoising autoencoder model
class ConvDenoiser(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, padding=0,  stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 16, 2, padding=0, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 32, 2, padding=0, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         return self.model(x)

# Instantiate the model
model = ConvDenoiser()
device = torch.device("cpu")
checkpoint = torch.load('denoise_25_rnd.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Define a transformation to preprocess the input image     
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/denoise', methods=['POST'])
def denoise():
    try:
        data = request.json
        drawing_data = data.get('drawingData')
        
        # Preprocess image data
        image_array = np.asarray(drawing_data, dtype=np.uint8).reshape(280, 280, 4)
        
        image_array = image_array[:, :, 3]  # only the alpha channel

        
        image_tensor = transform(image_array)
        image_tensor = image_tensor.view(1, 1, 28, 28).float()
       
        print("Processed image shape:", image_tensor.shape)
        denoised_image = model(image_tensor)
        
        print("Denoised image shape:", denoised_image.shape)
        denoised_image_pil = transforms.ToPILImage()(denoised_image[0])
        
        output_size = (280, 280)  # Adjust the size here
        denoised_image_pil = transforms.Resize(output_size)(denoised_image_pil)
        
        # Save the denoised image to a BytesIO object
        output_buffer = io.BytesIO()
        denoised_image_pil.save(output_buffer, format='PNG')
        output_data = base64.b64encode(output_buffer.getvalue()).decode()
       
        return jsonify({'outputImage': output_data})

    except Exception as e:
         print(f"An error occurred: {str(e)}")
         return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
