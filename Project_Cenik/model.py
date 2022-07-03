import tensorflow as tf
import tensorflow.keras.layers as nn
from tensorflow.keras import Model
import numpy as np
import tensor2tensor.layers.area_attention as area_attention


# Model architecture based on the paper
# We are implementing the multi-scale area attention together with the CNN architecture
# Area_key_mode='mean', area_value_mode='sum' different for paper since it is easily implemented with similar to ordinary attention

class AACNN(Model):
    def __init__(self, height=3, width=3, out_size=4):
        super(AACNN, self).__init__()
        self.height = height
        self.width = width
        # Two parallel convolutional layers with appropriate pooling and batch normalizations

        self.conv1 = nn.Conv2D(32, (3, 3), padding='same', data_format='channels_last', )
        self.conv1a = nn.Conv2D(16, (10, 2), padding='same', data_format='channels_last', )
        self.conv1b = nn.Conv2D(16, (2, 8), padding='same', data_format='channels_last', )
        self.conv2 = nn.Conv2D(32, (3, 3), padding='same', data_format='channels_last', )
        self.conv3 = nn.Conv2D(48, (3, 3), padding='same', data_format='channels_last', )
        self.conv4 = nn.Conv2D(64, (3, 3), padding='same', data_format='channels_last', )
        self.conv5 = nn.Conv2D(80, (3, 3), padding='same', data_format='channels_last', )
        self.conv6 = nn.Conv2D(128, (3, 3), padding='same', data_format='channels_last', )
        self.maxp = nn.MaxPool2D((2, 2))
        self.bn1a = nn.BatchNormalization(3)
        self.bn1b = nn.BatchNormalization(3)
        self.bn2 = nn.BatchNormalization(3)
        self.bn3 = nn.BatchNormalization(3)
        self.bn4 = nn.BatchNormalization(3)
        self.bn5 = nn.BatchNormalization(3)
        self.bn6 = nn.BatchNormalization(3)
        self.gap = nn.GlobalAveragePooling2D(data_format='channels_last')
        self.flatten = nn.Flatten(data_format='channels_last')
        self.fc = nn.Dense(out_size, activation='softmax')
        self.query = nn.Dense(20)
        self.key = nn.Dense(20)
        self.value = nn.Dense(20)

    def call(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = tf.nn.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = tf.nn.relu(xb)
        x = tf.concat([xa, xb], 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)
        q = x
        k = x
        v = x
        bias = None
        dropout_rate = 0.5

        # Multi-Scale Area Atention Mechanism with explicitly defined Key and Value parameters
        x = area_attention.dot_product_area_attention(
            q, k, v, bias, dropout_rate, None,
            save_weights_to=None,
            dropout_broadcast_dims=None,
            max_area_width=self.width,    # Fix the max area width
            max_area_height=self.height,  # Fix the max area height
            area_key_mode='mean',
            area_value_mode='sum',
            training=True)

        x = self.flatten(x)       # Fully connected layer to get the classifications
        x = self.fc(x)
        return x


if __name__ == '__main__':
    test = np.random.random((4, 40, 40, 1)).astype(np.float32)
    test = tf.convert_to_tensor(test)
    model = AACNN()
    y = model(test)
    s = tf.Session()
    print(s.run(y))