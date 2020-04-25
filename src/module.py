# coding:utf-8
# 网络基本模块

import tensorflow as tf
slim = tf.contrib.slim

# conv + BN + LeakRelu
def CBL(inputs, out_channels, kernel_size=3, stride=1, bn=True, relu=True, isTrain=True):
    '''
    inputs:输入tensor
    out_channels:输出的维度
    kernel_size:卷积核大小
    stride:步长
    bn:是否使用 batch_normalization
    relu:是否使用leak_relu激活
    isTrain:是否是训练, 训练会更新 bn 的滑动平均
    return:tensor
    ...
    普通卷积:
        input : [batch, height, width, channel]
        kernel : [height, width, in_channels, out_channels]
    '''
    # 补偿边角
    if stride > 1:
        inputs = padding_fixed(inputs, kernel_size)
    
    # 这里可以自定义激活方式, 默认 relu, 可以实现空洞卷积:rate 参数
    inputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride, 
                                                padding=('SAME' if stride == 1 else 'VALID'),
                                                activation_fn=None,
                                                normalizer_fn=None)
    
    if bn:
        # 如果想要提高稳定性，zero_debias_moving_mean设为True
        inputs = tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, updates_collections=None, 
                                                scale=False, is_training = isTrain)
    if relu:
        inputs = tf.nn.leaky_relu(inputs, alpha=0.1)
    return inputs

# leaky_relu 实现
def leaky_relu(inputs, alpha=0.1):
    '''
    leaky_relu 实现
    '''
    return tf.maximum(alpha*inputs, inputs)

# 边缘全零填充补偿卷积缺失
def padding_fixed(inputs, kernel_size):
    '''
    对tensor的周围进行全0填充
    '''
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start
    inputs = tf.pad(inputs, [[0,0], [pad_start, pad_end], [pad_start, pad_end], [0,0]])
    return inputs

# leaky_relu 实现
def leaky_relu(inputs, alpha=0.1):
    ''' 
    leaky_relu 实现
    '''
    return tf.maximum(inputs, inputs * alpha)

# yolo 残差模块实现
def yolo_res_block(inputs, kernel_num, isTrain):
    '''
    yolo的残差模块实现
    '''
    tmp = inputs
    net = CBL(inputs, kernel_num, 1, isTrain=isTrain)
    net = CBL(net, kernel_num*2, 3, isTrain=isTrain)
    net = net + tmp
    return net

# darknet53实现
def darknet53(inputs, isTrain=True):
    '''
    inputs:[N, 416, 416, 3]
    darknet53实现
    只有52个卷积层
    '''
    # ########## 第一阶段 ############
    # 先卷积两次
    net = CBL(inputs, 32, isTrain=isTrain)
    net = CBL(net, 64, stride=2, isTrain=isTrain)

    # 1个残差
    net = yolo_res_block(net, 32, isTrain=isTrain)
    net = CBL(net, 128, stride=2, isTrain=isTrain)

    # 两个残差
    for i in range(2):
        net = yolo_res_block(net, 64, isTrain=isTrain)
    net = CBL(net, 256, stride=2, isTrain=isTrain)

    # 8个残差
    for i in range(8):
        net = yolo_res_block(net, 128, isTrain=isTrain)
    # 得到第一阶段特征
    route_1 = net

    # ############## 第二阶段 ###############
    net = CBL(net, 512, stride=2, isTrain=isTrain)

    # 8个残差
    for i in range(8):
        net = yolo_res_block(net, 256, isTrain=isTrain)
    # 得到第二阶段特征
    route_2 = net

    # ############## 第三阶段 ###############
    net = CBL(net, 1024, stride=2, isTrain=isTrain)

    # 4个残差
    for i in range(4):
        net = yolo_res_block(net, 512, isTrain=isTrain)
    # 得到第三阶段特征
    route_3 = net

    return route_1, route_2, route_3


# YOLO 的卷积实现
def yolo_block(inputs, kernel_num, isTrain):
    '''
    yolo最后一段的卷积实现
    return:route,net, route比net少一个卷积, route用于与下一层特征进行拼接
    '''
    net = CBL(inputs, kernel_num, 1, isTrain=isTrain)
    net = CBL(net, kernel_num * 2, isTrain=isTrain)
    net = CBL(net, kernel_num, 1, isTrain=isTrain)
    net = CBL(net, kernel_num * 2, isTrain=isTrain)
    net = CBL(net, kernel_num, 1, isTrain=isTrain)
    route = net
    net = CBL(net, kernel_num * 2, isTrain=isTrain)
    return route, net

# yolo上采样模块
def yolo_upsample(inputs, out_shape):
    out_height, out_width = out_shape[1], out_shape[2]
    inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, (out_height, out_width))
    return inputs

# yolo的拼接模块
def yolo_concat(inputs1, inputs2, axis=3):
    return tf.concat([inputs1, inputs2], axis=axis)