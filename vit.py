###################################################################
# 可训练模式（常规训练测试模式,encoder结构中未加mask）
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms


# 上述的Naive形式为原始的patch构造embedding（NLP）方式
# 此操作为卷积构造，可看成用CNN+Transformer形式（当然更进一步的修改类似）
def image2emb_conv(image, kernel, stride):
    conv_output = F.conv2d(image, kernel, stride=stride)  # batch_size*outputChannel*height*width
    batch_size, outChannel, height, width = conv_output.shape
    patch_embedding = conv_output.reshape(batch_size, outChannel, height * width).transpose(-1, -2)
    # print(patch_embedding.shape)
    return patch_embedding


def make_token_embedding(patch_embeddding_conv):
    # print(patch_embeddding_conv.shape)
    # 即构造VIT输入[CLS embedding;N 个 patch embedding]or[CLS embedding;height * width embedding]
    # N :patch_num,D:model_dim(即Conv的outputChannel)
    # height * width == patch_num * patch_size ^ 2

    # 2: add prepared CLS token embedding(class embeddding)
    # 注意：同样可以不加CLS embedding 而在VIT输出使用average pooling得到最终的image presentation
    # 原文ViT是为了尽可能是模型结构接近原始的Transformer，所以采用了类似于BERT的做法，加入特殊字符
    cls_token_embedding = torch.randn(batch_size, 1, model_dim, requires_grad=True)
    token_embedding = torch.cat([cls_token_embedding, patch_embeddding_conv], dim=1)
    # print(token_embedding.shape)

    # 3: add position embedding(patch_num+1,model_dim):(N+1,D)，代码中为一维位置编码，EG：3x3共9个patch，patch编码为1到9
    # 也可用二维位置编码，EG：patch编码为11,12,13,21,22,23,31,32,33
    # 或也可用相对位置编码(eg:9相对1距离为8)，但各position embedding应展成embedding或用mask方式加入
    position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
    seq_len = token_embedding.shape[1]

    position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
    # print(position_embedding.shape)
    token_embedding += position_embedding

    return token_embedding


#########################################
# 4:pass embedding to Vit Model
class VitModel(nn.Module):
    def __init__(self):
        super(VitModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.num_layers = num_layers
        self.linear = nn.Linear(model_dim, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, token_embedding):
        transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.num_layers)
        x = transformer_encoder(token_embedding)
        cls_token_output = x[:, 0, :]  # Batch_size,位置，channel数目
        y = self.softmax(self.linear(cls_token_output))
        return y


# def train(model, dataset, lr, batch_size, num_epochs):
#     data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
#     for epoch in range(num_epochs):
#         losses = 0
#         for images in data_loader:
#             target = targets
#             outputs = model(input)
#             # print(outputs.shape)
#             loss = criterion(outputs, target)  # 训练集、测试集和标签的设定对模型效果影响很大
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses = losses + loss.item()
#         if (epoch + 1) % 5 == 0:
#             print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(losses / (data_loader.__len__())))


if __name__ == '__main__':
    # 通常在一个很大的数据集上预训练ViT，然后在下游任务相对小的数据集上微调，已有研究表明在分辨率更高的图片上微调比在在分辨率更低的图片上预训练效果更好
    image = Image.open("./cat.png").convert('RGB')
    image = transforms.Resize((450, 450))(image)  # 保持长宽比的resize方法
    # img = transforms.Resize((448,448))(img)  # 直接resize成正方形的方法
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    ############################################################
    # 1: make token_embedding attributes
    batch_size = 1
    imageChannel, width, height = image.shape[0], image.shape[1], image.shape[2]
    # print(imageChannel, width, height)
    image = image.unsqueeze(0)  # 拓展维度, 拓展batch_size那一维
    # print(image.shape)
    patch_size = 4
    model_dim = 8
    patch_depth = patch_size * patch_size * imageChannel  # one patch size of the image
    weight = torch.randn(model_dim, patch_depth)  # Conv2D:model_dim是outputChannel,patch_depth是结果conv size*intputChannel
    kernel = weight.reshape((-1, imageChannel, patch_size, patch_size))

    patch_embeddding_conv = image2emb_conv(image, kernel, patch_size)

    max_num_token = height*width + 1  # height*width == patch_num*patch_size^2, max_num_token >= height*width(patch_num*patch_size^2) + 1(class embedding)

    token_embedding = make_token_embedding(patch_embeddding_conv)

    ##############################
    # define model attributes
    nhead = 8
    num_layers = 6
    num_classes = 10

    model = VitModel()
    label = torch.randint(10, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    output = model(token_embedding)
    loss = criterion(output, label)
    print(loss)
