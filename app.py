import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# 必须和训练时的模型定义一样（这里只复制 ModelA，你们也可以改成加载效果最好的）
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# 加载模型（使用刚刚训练好的 ModelA.pth）
model = ModelA()
model.load_state_dict(torch.load('ModelA.pth', map_location=torch.device('cpu')))
model.eval()

# 图像预处理（和训练时一致）
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

st.title("✍️ 手写数字识别")
st.write("上传一张手写数字图片（黑底白字），模型会预测是 0-9 中的哪个数字。")

uploaded = st.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="上传的图片", width=150)
    # 预处理
    img_tensor = transform(img).unsqueeze(0)  # 加 batch 维度
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
        prob = torch.exp(output).max().item()
    st.success(f"**预测结果：{pred}**   (置信度: {prob:.2f})")