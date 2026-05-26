[README.md](https://github.com/user-attachments/files/28261865/README.md)
# mnist-handwriting-recognition# 手写数字识别 - MNIST 深度学习项目

## 项目简介
基于 PyTorch 和 Streamlit 的手写数字识别系统。使用三个不同结构的 CNN 模型在 MNIST 数据集上训练，提供 Web 界面供用户上传手写数字图片进行识别。

## 主要功能
- 训练三个 CNN 模型（简单CNN、深层CNN、BatchNorm CNN）并进行对比
- 评估模型准确率、混淆矩阵
- 通过 Streamlit 网页上传图片，实时预测数字并显示置信度

## 环境要求
- Python 3.8 或更高版本
- 依赖库：见 requirements.txt

## 快速开始
公网部署链接：https://mnist-handwriting-recognition-mskolg6qheh2g4cpvwvpft.streamlit.app/
### 1. 安装依赖
pip install -r requirements.txt
或手动安装：
pip install torch torchvision streamlit scikit-learn numpy matplotlib pillow

### 2. 训练模型
python train.py
训练完成后会生成 ModelA.pth, ModelB.pth, ModelC.pth 三个模型文件，并输出测试准确率。

### 3. 评估模型（生成混淆矩阵）
python evaluate.py
会输出每个模型的准确率，并生成 ModelA_cm.png, ModelB_cm.png, ModelC_cm.png 混淆矩阵图。

### 4. 启动 Web 演示界面
streamlit run app.py
如果提示 streamlit: command not found，请使用：
python3 -m streamlit run app.py
浏览器将自动打开 http://localhost:8501，上传手写数字图片（建议黑底白字），即可看到预测结果和置信度。

## 项目文件结构
.
├── train.py               # 训练三个模型
├── evaluate.py            # 评估模型并生成混淆矩阵
├── app.py                 # Streamlit Web 应用
├── save_test_images.py    # （可选）提取测试集图片
├── requirements.txt       # 依赖库列表
├── README.md              # 本文件
├── ModelA.pth             # 训练好的模型权重（运行后生成）
├── ModelB.pth
├── ModelC.pth
└── mnist_test_images/     # 保存的测试集图片（可选）

## 三个模型对比结果（3个epoch）
| 模型 | 结构特点 | 测试准确率 |
|------|----------|------------|
| ModelA | 2层卷积 + Dropout | 98.96% |
| ModelB | 3层卷积 + 自适应池化 | 98.32% |
| ModelC | 2层卷积 + BatchNorm | 98.81% |

## 常见问题
- 预测置信度偏低：训练轮数较少（仅3轮），增加 EPOCHS 至10~20可显著提升。
- streamlit: command not found：使用 python3 -m streamlit run app.py。
- 图片识别错误：请确保上传的图片是黑底白字，数字清晰居中。

## 数据集
MNIST 手写数字数据集，包含 60000 张训练图片和 10000 张测试图片，图片大小为 28×28 灰度图。数据集会在第一次运行 train.py 时自动下载。

## 团队分工
- 成员李嘉艺：数据加载与预处理
- 成员沈怿成：模型设计（三个CNN结构）
- 成员邓涵予：训练流程与模型保存
- 成员徐若瑜：模型评估与可视化（准确率、混淆矩阵）
- 成员沈歆：系统实现（Streamlit界面）、云端部署、项目统筹与汇总

## 参考资料
- PyTorch 官方文档：https://pytorch.org/
- Streamlit 官方文档：https://streamlit.io/
- MNIST 数据集：http://yann.lecun.com/exdb/mnist/
