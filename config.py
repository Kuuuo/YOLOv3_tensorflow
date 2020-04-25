# coding:utf-8
# 配置文件

# ############# 基本配置 #############
class_num = 25
anchors = 13,20, 19,37, 23,38, 29,32, 26,38, 29,40, 34,35, 39,44, 67,96
model_path = "./checkpoint/"
model_name = "model"
name_file = './data/train.names'

# ############# 日志 #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# ############## 训练 ##############
train_file = './data/train.txt'
batch_size = 4
multi_scale_img = True     # 多尺度缩放图片训练
total_epoch = 300       # 一共训练多少 epoch
save_step = 1000        # 多少步保存一次

use_iou = True      # 计算损失时, 以iou作为衡量标准, 否则用 giou
ignore_thresh = 0.5     # 与真值 iou / giou 小于这个阈值就认为没有预测物体

# 学习率配置
lr_init = 1e-4                      # 初始学习率
lr_lower = 1e-6                 # 最低学习率
lr_type = 'piecewise'   # 学习率类型 'exponential', 'piecewise', 'constant'
piecewise_boundaries = [100, 300]   # 单位:epoch, for piecewise
piecewise_values = [lr_init, 5e-5, 1e-5]

# 优化器配置
optimizer_type = 'momentum' # 优化器类型
momentum = 0.9          # 动量



# ############## 测试 ##############
score_thresh = 0.5      # 少于这个分数就忽略
iou_thresh = 0.5            # iou 大于这个值就认为是同一个物体
max_box = 50                # 物体最多个数
val_dir = "./test_pic"  # 测试文件夹, 里面存放测试图片
save_img = True             # 是否保存测试图片
save_dir = "./save"         # 图片保存路径
width = 416                     # resize 的图片宽
height = 416                    # resize 的图片高


