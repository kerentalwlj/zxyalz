import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split
import os
import cv2
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.optim import Adam,lr_scheduler
import random
import torch.optim as optim
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from collections import namedtuple
#这是一个为wassertein损失函数增加注意力模块的注意力模块定义

#这是一个判断图片是否有效的函数

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False
#这是一个获取有效的图片的函数
def get_valid_files(root):
    valid_files = []
    for subdir, _, files in os.walk(root):
        for file in files:
            filepath = os.path.join(subdir, file)
            if is_valid_image(filepath):
                valid_files.append(filepath)
    return valid_files

#这是更好的数据加载器，能够只加载有效的图像而忽略无效的图像
class ValidImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = [(path, self.class_to_idx[os.path.basename(os.path.dirname(path))]) for path in
                        get_valid_files(root)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        sample = default_loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, label
scaler = GradScaler()
import torch.nn as nn
import torchvision.models as models

class AttentionWeights(nn.Module):
    def __init__(self, in_dim):
        super(AttentionWeights, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 添加这一行
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        sorted_indices = torch.argsort(x, descending=True)
        return x, sorted_indices


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature = None
        self.gradient = None

        # 查找最后一个卷积层
        target_layer = self._find_last_conv_layer(self.model.encoder)

        if target_layer is None:
            raise ValueError("Cannot find a convolutional layer in the model.")

        target_layer.register_forward_hook(self.save_feature)
        target_layer.register_backward_hook(self.save_gradient)

    def _find_last_conv_layer(self, module):
        """
        递归查找给定模块中的最后一个卷积层。
        """
        # 检查此模块是否是卷积层
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            return module

        # 递归查找子模块
        for child in reversed(list(module.children())):
            result = self._find_last_conv_layer(child)
            if result:
                return result
        return None

    def save_feature(self, module, input, output):
        self.feature = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, x, idx=None):
        self.model.eval()
        output = self.model(x)
        if idx is None:
            idx = output.argmax(dim=1)

        one_hot = torch.zeros_like(output).to(x.device)
        one_hot.scatter_(1, idx.view(-1, 1), 1)

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        gradient_channel_mean = self.gradient.mean(dim=[2, 3])
        cam = (self.feature * gradient_channel_mean[..., None, None]).sum(dim=1)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()

        return cam
class DomainAdaptationModel(nn.Module):
    def __init__(self, in_channel=3, num_classes=2):
        super(DomainAdaptationModel, self).__init__()
        self.encoder = models.resnet152(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))
        self.flatten = nn.Flatten()
        self.attention_net = AttentionWeights(2048)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            # 2048到1024的全连接层
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512, bias=True),  # 1024到512的全连接层
            nn.Dropout(0.6),
            nn.Linear(512, 256, bias=True),
            nn.Dropout(0.6),
            nn.BatchNorm1d(256),  # 512到256的全连接层
            nn.Linear(256, num_classes, bias=True)  # 256到分类类别数的全连接层
        )

    def forward(self, input):  # 前向传播函数
        x = self.encoder(input)
        x = self.flatten(x)
        attention_weights, _ = self.attention_net(x)
        x = x * attention_weights
        output = self.classifier(x)
        return output  # 返回分类输出

    def compute_attention(self, encoded_source, encoded_target):
        encoded_source = encoded_source.view(encoded_source.size(0), -1)  # 添加这一行
        encoded_target = encoded_target.view(encoded_target.size(0), -1)  # 添加这一行
        differences = encoded_source - encoded_target
        attention_weights, sorted_indices = self.attention_net(differences)
        return attention_weights, sorted_indices

    def balanced_sliced_wasserstein_distance(self, encoded_source, encoded_target, num_projections=5000):
        device = encoded_source.device
        projections = torch.randn((encoded_source.size(1), num_projections)).to(device)
        projections /= torch.norm(projections, dim=0, keepdim=True)

        encoded_source = encoded_source.squeeze(-1).squeeze(-1)
        encoded_target = encoded_target.squeeze(-1).squeeze(-1)
        projected_source = torch.mm(encoded_source, projections)
        projected_target = torch.mm(encoded_target, projections)
        sorted_source, _ = torch.sort(projected_source, dim=0)
        sorted_target, _ = torch.sort(projected_target, dim=0)
        swd = torch.mean(torch.abs(sorted_source - sorted_target), dim=0).mean()

        return swd

image_dim=64
class Grayscale(object):
    def __call__(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(image)

class Denoise(object):
    def __call__(self, image):
        image = cv2.fastNlMeansDenoisingColored(np.array(image), None, 5, 5, 5, 21)
        return Image.fromarray(image)


class HistEqualization(object):
    def __call__(self, image):
        img_yuv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(image)
# 3. Data loading
transform1 = transforms.Compose([
    # Denoise(),  # 去噪
    # HistEqualization(),  # 直方图均衡化
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    # Denoise(),  # 去噪
    # HistEqualization(),  # 直方图均衡化
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# image_dim=64
source_dataset = ValidImageFolder(root='./oral_dataset/source_domain', transform=transform1)
target_dataset = ValidImageFolder(root='./oral_dataset/target_domain', transform=transform2)
source_dataloader = DataLoader(source_dataset, batch_size=128, shuffle=True,pin_memory=True,drop_last=True)
target_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=True,pin_memory=True,drop_last=True)
#####################################################划分验证集的数据
validation_fraction = 0.1
num_source_samples = len(source_dataset)
num_target_samples = len(target_dataset)
num_source_valid = int(validation_fraction * num_source_samples)
num_target_valid = int(validation_fraction * num_target_samples)
source_train_dataset, source_valid_dataset = random_split(source_dataset, [num_source_samples - num_source_valid, num_source_valid])
target_train_dataset, target_valid_dataset = random_split(target_dataset, [num_target_samples - num_target_valid, num_target_valid])
source_train_dataloader = DataLoader(source_train_dataset, batch_size=128, shuffle=True,pin_memory=True,drop_last=True)
source_valid_dataloader = DataLoader(source_valid_dataset, batch_size=128, shuffle=True,pin_memory=True,drop_last=True)
target_train_dataloader = DataLoader(target_train_dataset, batch_size=128, shuffle=True,pin_memory=True,drop_last=True)
target_valid_dataloader = DataLoader(target_valid_dataset, batch_size=128, shuffle=True,pin_memory=True,drop_last=True)
# 4. Training, evaluation, and visualization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
######################验证集合的评估函数
def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
def progressive_sampling(initial_percentage, increase_percentage, max_percentage, epoch):
    percentage = min(initial_percentage + epoch * increase_percentage, max_percentage)#渐进引入目标域数据
    return percentage
def mix_training_step(source_data, source_labels, target_data, model, teacher_model, criterion, confident_samples_mask):
    source_pred = model(source_data)
    classification_loss = criterion(source_pred, source_labels)
    with torch.no_grad():
        soft_target = teacher_model(target_data)
    target_pred = model(target_data)
    distillation_loss = criterion(target_pred[confident_samples_mask], soft_target.argmax(1)[confident_samples_mask])

    source_encoded = model.encoder(source_data)
    target_encoded = model.encoder(target_data)
    # attention_weights, _ = model.compute_attention(source_encoded, target_encoded)
    swd = model.balanced_sliced_wasserstein_distance(source_encoded[confident_samples_mask], target_encoded[confident_samples_mask])
    total_loss = 2*classification_loss + 0*distillation_loss + 10*swd

    return total_loss
def find_high_confidence_samples(data, model, threshold=0.95):
    with torch.no_grad():
        preds = model(data)
        probs = F.softmax(preds, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        mask = confidence >= threshold
    return mask
def train(model, source_dataloader, target_dataloader, num_epochs=50, save_path='./models'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

    initial_percentage = 0.1
    increase_percentage = 0.05
    max_percentage = 1.0

    best_accuracy = 0.0
    pretrain_model_path = os.path.join(save_path, "pretrained_source_model.pth")

    # Pretraining on the source domain
    if not os.path.exists(pretrain_model_path):
        for epoch in range(num_epochs):
            print("正在源域上预训练")
            for source_data, source_labels in source_dataloader:
                source_data, source_labels = source_data.to(device), source_labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    source_pred = model(source_data)
                    classification_loss = criterion(source_pred, source_labels)
                    total_loss = classification_loss
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()

            valid_accuracy_source = evaluate_accuracy(model, source_valid_dataloader)
            print(
                f'classification_loss={classification_loss.item()} / Validation Accuracy on Source Domain after Epoch [{epoch + 1}/{num_epochs}]: {valid_accuracy_source}%')

            if valid_accuracy_source >= 97.0:
                torch.save(model.state_dict(), pretrain_model_path)
                print(f"Source domain pretraining model saved to {pretrain_model_path}")

    # Load the pretrained model as teacher model
    teacher_model = type(model)()
    teacher_model.load_state_dict(torch.load(pretrain_model_path))
    teacher_model.to(device)
    teacher_model.eval()
    model.load_state_dict(torch.load(pretrain_model_path))
    print("开始进行域适应训练")

    for epoch in range(num_epochs):
        model.train()  # 确保模型处于训练模式
        current_percentage = progressive_sampling(initial_percentage, increase_percentage, max_percentage, epoch)
        num_target_samples = max(2, int(len(target_dataloader.dataset) * current_percentage))  # 确保至少是2
        target_indices = torch.randperm(len(target_dataloader.dataset))[:num_target_samples]
        target_subset_sampler = torch.utils.data.SubsetRandomSampler(target_indices)
        target_subset_loader = torch.utils.data.DataLoader(target_dataloader.dataset,
                                                           batch_size=target_dataloader.batch_size,
                                                           sampler=target_subset_sampler)

        for (source_data, source_labels), (target_data, _) in zip(source_dataloader, target_subset_loader):
            # 通过比较batch sizes 来处理不同大小的数据集
            min_batch_size = min(source_data.shape[0], target_data.shape[0])
            if min_batch_size == 1:
                continue  # 如果有一个batch size为0，则跳过

            source_data, source_labels = source_data.to(device)[:min_batch_size], source_labels.to(device)[
                                                                                  :min_batch_size]
            target_data = target_data.to(device)[:min_batch_size]
            optimizer.zero_grad()

            with autocast():
                source_pred = model(source_data)
                classification_loss = criterion(source_pred, source_labels)
                with torch.no_grad():
                    soft_target = teacher_model(target_data)

                target_pred = model(target_data)
                distillation_loss = criterion(target_pred, soft_target.argmax(dim=1))

                source_encoded = model.encoder(source_data)
                target_encoded = model.encoder(target_data)
                # attention_weights, _ = model.compute_attention(source_encoded, target_encoded)
                swd = model.balanced_sliced_wasserstein_distance(source_encoded, target_encoded)

                target_probs = F.softmax(target_pred, dim=1)
                target_confidence, _ = torch.max(target_probs, dim=1)
                confident_samples_mask = target_confidence >= 1
                confident_samples_count = confident_samples_mask.sum().item()

                # 使用混合训练策略当有足够自信的样本时
                if confident_samples_count > 3:
                    total_loss = 10 * swd + 2 * classification_loss + 0.5 * distillation_loss  # 调整损失函数权重
                else:
                    total_loss = 10 * swd + 2 * classification_loss + 0.5 * distillation_loss

                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()

        scheduler.step()  # 在每个epoch结束后更新学习率

        valid_accuracy_source = evaluate_accuracy(model, source_valid_dataloader)
        valid_accuracy_target = evaluate_accuracy(model, target_valid_dataloader)
        print(f"total loss=swd:{swd}+distillation loss{distillation_loss}+classification loss:{classification_loss}")
        print(f"Epoch:{epoch + 1}")
        print(f'Validation Accuracy on Source Domain : {valid_accuracy_source:.2f}%')
        print(f'Validation Accuracy on Target Domain : {valid_accuracy_target:.2f}%')

        # Save model if it has the best accuracy on target validation set so far
        if valid_accuracy_target > best_accuracy:
            best_accuracy = valid_accuracy_target
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)

    print("域适应训练完成")
    # Load the best model to evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_accuracy_source = evaluate_accuracy(model, source_valid_dataloader)
    test_accuracy_target = evaluate_accuracy(model, target_valid_dataloader)
    print(f'Test Accuracy on Source Domain: {test_accuracy_source:.2f}%')
    print(f'Test Accuracy on Target Domain: {test_accuracy_target:.2f}%')

    return model
def evaluate(model, dataloader, domain_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on {domain_name}: {accuracy}%')


def visualize_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)

            encoded = model.encoder(data).cpu().numpy()
            features.append(encoded)
            labels.append(label.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='jet')
    plt.colorbar()
    plt.show()


def display_gradcam(img, heatmap, alpha=0.6):
    # 确保输入图像是uint8且范围在0-255之间
    if img.dtype != np.uint8:
        if img.max() > 1:
            img = (img / 255).astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8)

    # 如果热图是二维的，扩展为三维
    if heatmap.ndim == 2:
        heatmap = np.expand_dims(heatmap, axis=-1)

    # 确保热图范围在0-1之间
    heatmap = np.clip(heatmap, 0, 1)

    # 使用cv2的函数将热图转换为JET颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # 转换为RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 根据alpha组合原始图像和热图
    heatmap_combined = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # 创建新的图形并显示组合后的图像
    plt.figure()
    plt.imshow(heatmap_combined)
    plt.axis('off')
    plt.show()

# Initialize model and start training
model = DomainAdaptationModel(2)
model = model.to(device)
def overlay_gradcam_on_image(image, cam):
    # Resize the cam to the size of the image using cubic interpolation
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.uint8(255 * cam_resized)

    # Convert the grayscale cam to colormap
    cam_colored = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    # Convert image from [0,1] float to [0,255] uint8 format
    image_uint8 = np.uint8(255 * image)

    # Overlay the image with the colored cam and normalize with adjusted weights
    overlaid = cv2.addWeighted(image_uint8, 0.8, cam_colored, 0.2, 0)
    overlaid = overlaid / 255.0

    return overlaid


def show_random_images_with_gradcam(dataloader, model, num_images=16, save_dir='gradcam_images'):
    # Create GradCAM object
    gradcam = GradCAM(model)

    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Randomly pick num_images batches
    random_batches = random.sample(list(dataloader), num_images)

    for i, (images, labels) in enumerate(random_batches):
        # For each batch, we randomly pick an image
        idx = random.choice(range(images.shape[0]))

        # Put the model in evaluation mode and move the image to the right device
        model.eval()
        image = images[idx].unsqueeze(0).to(device)
        label = labels[idx]

        # Compute GradCAM mask
        cam = gradcam(image)
        cam = cam.cpu().detach().numpy()

        # Prepare original image and overlaid image
        original_image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        overlaid_image = overlay_gradcam_on_image(original_image, cam.squeeze())

        # Create a figure with 2 subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Display original image
        ax[0].imshow(original_image)
        ax[0].set_title(f"Original Image {i + 1}")
        ax[0].axis('off')

        # Display GradCAM overlaid image
        ax[1].imshow(overlaid_image)
        ax[1].set_title(f"GradCAM Image {i + 1} - Label: {label}")
        ax[1].axis('off')

        # Save the figure to the specified directory
        plt.savefig(os.path.join(save_dir, f'combined_image_{i + 1}.png'))

        # Optionally, display the image on screen
        # plt.show()

        # Clear the current figure to free memory
        plt.clf()
        plt.close('all')


# model.load_state_dict(torch.load('models/best_model.pth'))
model.to(device)

train(model, source_dataloader, target_dataloader, num_epochs=500)
evaluate(model, source_dataloader, "Source Domain")
evaluate(model, target_dataloader, "Target Domain")
visualize_features(model, target_dataloader)
# Extract features
gradcam = GradCAM(model)
show_random_images_with_gradcam(source_dataloader, model)
evaluate_accuracy(model, source_dataloader)
evaluate_accuracy(model, target_dataloader)


