
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# 定义 AlexNet 模型
def alexnet(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(96, kernel_size=(3, 3), strides=1, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = alexnet()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, batch_size=8, epochs=100, validation_data=(x_test, y_test))

# 可视化训练过程
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy}")
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 选择模型的某一层作为特征提取器
# 例如，我们可以使用第一个全连接层
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-6].output)

# 提取训练集和测试集的特征
features_train = feature_extractor.predict(x_train)
features_test = feature_extractor.predict(x_test)

# 使用 t-SNE 对特征进行降维
tsne = TSNE(n_components=2, random_state=0)
tsne_result_train = tsne.fit_transform(features_train)
tsne_result_test = tsne.fit_transform(features_test)

# 可视化函数
def plot_tsne(tsne_result, labels, title='t-SNE'):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=np.argmax(labels, axis=1), alpha=0.6)
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

# 可视化训练集的 t-SNE 结果
plot_tsne(tsne_result_train, y_train, 't-SNE of CIFAR-10 Training Data')

# 可视化测试集的 t-SNE 结果
plot_tsne(tsne_result_test, y_test, 't-SNE of CIFAR-10 Test Data')
