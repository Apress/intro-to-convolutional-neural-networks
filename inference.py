import torch
import sys
import re
import base64
import io
from flask import jsonify
from PIL import Image
from torchvision import models
from torchvision import transforms
from flask import Flask, escape, request

net = models.shufflenet_v2_x1_0(pretrained=True)
net.eval()

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


def infer(img):
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = net(batch_t)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    result = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
    return result

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer_image():
    image = str(request.form['image'])
    imgdata = base64.b64decode(image)
    image = Image.open(io.BytesIO(imgdata))
    return jsonify(infer(image))

app.run(port=5000)
