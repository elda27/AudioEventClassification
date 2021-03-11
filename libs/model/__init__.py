import tensorflow as tf
from libs.layers.conv_regressor import ConvRegressor
from libs.layers.resnet import (ResNet18, ResNet34, ResNet50, ResNet101)
from libs.layers.cbam_resnet import (ResNetCBAM18, ResNetCBAM34, ResNetCBAM50, ResNetCBAM101)
from libs.layers.conv_attention_resnet import (
    AttentionResNet18, AttentionResNet34, AttentionResNet50, AttentionResNet101,
    AxialAttentionResNet18, AxialAttentionResNet34, AxialAttentionResNet50, AxialAttentionResNet101,
)

try:
    from libs.layers.bigbird.bigbird_encoder import BigbirdEncoder
    from libs.layers.bigbird.pretrain_bigbird_encoder import PretrainingBigbirdEncoder
    ENABLE_BIGBIRD = True
except:
    ENABLE_BIGBIRD = False

from libs.layers.reformer.reformer_encoder import ReformerEncoder

from enum import Enum


class NetworkType(Enum):
    # 2D convolutional neural network
    CNN = "cnn"
    Resnet = "resnet"
    Resnet18 = "resnet18"
    Resnet34 = "resnet34"
    Resnet50 = "resnet50"
    Resnet101 = "resnet101"
    ResnetCBAM = "resnet_cbam"
    ResnetCBAM18 = "resnet_cbam18"
    ResnetCBAM34 = "resnet_cbam34"
    ResnetCBAM50 = "resnet_cbam50"
    ResnetCBAM101 = "resnet_cbam101"
    Attention = "attention"
    Attention18 = "attention18"
    Attention34 = "attention34"
    Attention50 = "attention50"
    Attention101 = "attention101"
    AxialAttention = "axial-attention"
    AxialAttention18 = "axial-attention18"
    AxialAttention34 = "axial-attention34"
    AxialAttention50 = "axial-attention50"
    AxialAttention101 = "axial-attention101"
    HourGlass = "hourglass"

    # 1D BERT based network
    BigBird = "bigbird"
    BigBird = "bigbird-pretrain"
    Reformer = "reformer"


NETWORK_DICT = {}


def register(network_type: NetworkType):
    def _(fn):
        NETWORK_DICT[network_type] = fn
        return fn
    return _


def create_model(network_type: NetworkType, n_class: int, **kwargs):
    if network_type in NETWORK_DICT:
        return NETWORK_DICT[network_type](n_class, **kwargs)
    else:
        raise NotImplementedError()


# =============================================================================
# Normal CNN
# =============================================================================

@register(NetworkType.CNN)
def create_cnn_model(n_class: int):
    return tf.keras.Sequential([
        ConvRegressor([8, 16, 32, 64, 128, 256], kernel_size=7),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


# =============================================================================
# ResNet
# =============================================================================
@register(NetworkType.Resnet)
@register(NetworkType.Resnet34)
def create_resnet34_model(n_class: int):
    return tf.keras.Sequential([
        ResNet34(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.Resnet18)
def create_resnet18_model(n_class: int):
    return tf.keras.Sequential([
        ResNet18(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.Resnet50)
def create_resnet50_model(n_class: int):
    return tf.keras.Sequential([
        ResNet50(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.Resnet101)
def create_resnet101_model(n_class: int):
    return tf.keras.Sequential([
        ResNet101(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])

# =============================================================================
# ResNet CBAM (Convolutional Block Attention Module)
# =============================================================================


@register(NetworkType.ResnetCBAM)
@register(NetworkType.ResnetCBAM34)
def create_resnet34_cbam_model(n_class: int):
    return tf.keras.Sequential([
        ResNetCBAM34(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.ResnetCBAM18)
def create_resnet18_cbam_model(n_class: int):
    return tf.keras.Sequential([
        ResNetCBAM18(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.ResnetCBAM50)
def create_resnet50_cbam_model(n_class: int):
    return tf.keras.Sequential([
        ResNetCBAM50(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.ResnetCBAM101)
def create_resnet101_cbam_model(n_class: int):
    return tf.keras.Sequential([
        ResNetCBAM101(kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


# =============================================================================
# ResNet Attention Network
# =============================================================================

@register(NetworkType.Attention)
@register(NetworkType.Attention34)
def create_attention34_model(n_class: int):
    return tf.keras.Sequential([
        AttentionResNet34(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.Attention18)
def create_attention18_model(n_class: int):
    return tf.keras.Sequential([
        AttentionResNet18(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.Attention50)
def create_attention50_model(n_class: int):
    return tf.keras.Sequential([
        AttentionResNet50(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.Attention191)
def create_attention101_model(n_class: int):
    return tf.keras.Sequential([
        AttentionResNet101(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


# =============================================================================
# ResNet Axial Attention Network
# =============================================================================

@register(NetworkType.AxialAttention)
@register(NetworkType.AxialAttention34)
def create_axial_attention34_model(n_class: int):
    return tf.keras.Sequential([
        AxialAttentionResNet34(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.AxialAttention18)
def create_axial_attention34_model(n_class: int):
    return tf.keras.Sequential([
        AxialAttentionResNet18(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.AxialAttention50)
def create_axial_attention34_model(n_class: int):
    return tf.keras.Sequential([
        AxialAttentionResNet50(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


@register(NetworkType.AxialAttention101)
def create_axial_attention34_model(n_class: int):
    return tf.keras.Sequential([
        AxialAttentionResNet101(kernel_size=1),
        tf.keras.layers.Conv2D(1024, kernel_size=7, strides=2, activation=tf.nn.relu),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_class),
    ])


# =============================================================================
# Bigbird encoder
# =============================================================================
@register(NetworkType.BigBird)
def create_bigbird(n_class: int, **kwargs):
    return tf.keras.Sequential([
        BigbirdEncoder(**kwargs),
        tf.keras.layers.Dense(n_class)
    ])

# =============================================================================
# Reformer encoder
# =============================================================================


@register(NetworkType.Reformer)
def create_reformer(n_class: int, **kwargs):
    return tf.keras.Sequential([
        ReformerEncoder(**kwargs),
        tf.keras.layers.Dense(n_class)
    ])
