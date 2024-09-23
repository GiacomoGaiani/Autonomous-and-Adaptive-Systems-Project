import tensorflow as tf
from tensorflow.keras import layers

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        x = self.bn1(self.conv1(x), training=training)
        x = self.relu(x)
        x = self.bn2(self.conv2(x), training=training)
        x += residual
        return self.relu(x)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.resblock1 = ResidualBlock(out_channels)
        self.resblock2 = ResidualBlock(out_channels)

    def call(self, x, training=False):
        x = self.bn1(self.conv1(x), training=training)
        x = self.pool(x)
        x = self.resblock1(x, training=training)
        x = self.resblock2(x, training=training)
        return x

    def get_config(self):
        config = super(ConvolutionalBlock, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ValueNetwork(tf.keras.Model):
    def __init__(self, input_shape, name=None, dtype=None, **kwargs):
        super(ValueNetwork, self).__init__(name=name, dtype=dtype, **kwargs)
        self.input_shape = input_shape
        self.conv_block1 = ConvolutionalBlock(input_shape[-1], 16)
        self.conv_block2 = ConvolutionalBlock(16, 32)
        self.conv_block3 = ConvolutionalBlock(32, 32)

        self.feed_forward = layers.Dense(512)
        self.value_head = layers.Dense(1)

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32) / 255.0
        x = self.conv_block1(x, training=training)
        x = self.conv_block2(x, training=training)
        x = self.conv_block3(x, training=training)

        x = tf.nn.relu(x)
        x = tf.reshape(x, [x.shape[0], -1])
        x = tf.nn.relu(self.feed_forward(x))
        return self.value_head(x)

    def get_config(self):
        config = super(ValueNetwork, self).get_config()
        config.update({
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop('input_shape')
        return cls(input_shape=input_shape, **config)


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, action_space, name=None, dtype=None, **kwargs):
        super(PolicyNetwork, self).__init__(name=name, dtype=dtype, **kwargs)
        self.input_shape = input_shape
        self.action_space = action_space
        self.conv_block1 = ConvolutionalBlock(input_shape[-1], 16)
        self.conv_block2 = ConvolutionalBlock(16, 32)
        self.conv_block3 = ConvolutionalBlock(32, 32)

        self.feed_forward = layers.Dense(512)
        self.policy_head = layers.Dense(action_space)
        self.softmax_activation = layers.Softmax(axis=-1)

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32) / 255.0
        x = self.conv_block1(x, training=training)
        x = self.conv_block2(x, training=training)
        x = self.conv_block3(x, training=training)

        x = tf.nn.relu(x)
        x = tf.reshape(x, [x.shape[0], -1])
        x = tf.nn.relu(self.feed_forward(x))
        return self.softmax_activation(self.policy_head(x))

    def get_config(self):
        config = super(PolicyNetwork, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'action_space': self.action_space
        })
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop('input_shape')
        action_space = config.pop('action_space')
        return cls(input_shape=input_shape, action_space=action_space, **config)


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, input_shape, action_space, name=None, dtype=None, **kwargs):
        super(ActorCriticNetwork, self).__init__(name=name, dtype=dtype, **kwargs)
        self.input_shape = input_shape
        self.action_space = action_space

        self.conv_block1 = ConvolutionalBlock(input_shape[-1], 16)
        self.conv_block2 = ConvolutionalBlock(16, 32)
        self.conv_block3 = ConvolutionalBlock(32, 32)

        self.actor_feed_forward = layers.Dense(512)
        self.policy_head = layers.Dense(action_space)
        self.softmax_activation = layers.Softmax(axis=-1)

        self.critic_feed_forward = layers.Dense(512)
        self.value_head = layers.Dense(1)

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32) / 255.0
        x = self.conv_block1(x, training=training)
        x = self.conv_block2(x, training=training)
        x = self.conv_block3(x, training=training)

        x = tf.nn.relu(x)
        x = tf.reshape(x, [x.shape[0], -1])

        policy = tf.nn.relu(self.actor_feed_forward(x))
        policy = self.softmax_activation(self.policy_head(policy))

        value = tf.nn.relu(self.critic_feed_forward(x))
        value = self.value_head(value)

        return policy, value

    def get_config(self):
        config = super(ActorCriticNetwork, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'action_space': self.action_space
        })
        return config