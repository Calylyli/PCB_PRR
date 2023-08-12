# PCB描述

PCB（Part-level Convolutional Baseline）模型是行人重识别任务的经典模型，它通过对输入图像进行均匀分划，构建出包含图像各部分特征的卷积描述子，用于后续行人检索。此外，原论文提出了RPP（Refined Part Pooling）方法对离群点所属区域进行重新分配，进一步提高了区域内特征的一致性，有效提升了PCB模型的精度。

如下为MindSpore使用DukeMTMC-reID数据集对PCB+RPP进行训练的示例。

## 论文

1. [论文](https://arxiv.org/pdf/1711.09349.pdf)：Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, Shengjin Wang."Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)"

# 模型架构

PCB的总体网络架构如下：
[链接](https://arxiv.org/pdf/1711.09349.pdf)

# 数据集

使用的数据集：[DukeMTMC-reID](http://vision.cs.duke.edu/DukeMTMC/)

- 数据集组成：
    - 训练集：包含702个行人的16522个RGB图像
    - 测试集：
        -query set: 包含702个行人的2228个RGB图像
        -gallery set：包含1110个行人的17661个RGB图像
- 下载数据集。目录结构如下：
```text
├─ DukeMTMC-reID
 │
 ├─bounding_box_test
 │
 └─bounding_box_train
 │
 └─query
```

# 环境要求

- 硬件(Ascend)
    - 准备Ascend910处理器搭建硬件环境。
- 框架
    - MindSpore=2.0
# 脚本说明

## 脚本及样例代码

```text
.
└──PCB
  ├── README.md
  ├── config                              # 参数配置
    ├── base_config.yaml
    ├── train_PCB_duke.yaml
      ├── train_PCB.yaml
      ├── finetune_PCB.yaml
    ├── train_RPP_duke
      ├── train_PCB.yaml
      ├── train_RPP.yaml
    ├── eval_PCB_duke.yaml
    ├── eval_RPP_duke.yaml
    ├── infer_310_config.yaml             # 用于模型导出成mindir、310推理的配置文件
  ├── scripts
    ├── run_standalone_train.sh           # 启动单卡训练脚本
    ├── run_distribute_eval.sh            # 启动八卡训练脚本
    ├── run_eval.sh                       # 启动评估脚本
    ├── run_infer_310.sh                  # 启动310推理
  ├── src
    ├── dataset.py                         # 数据预处理
    ├── eval_callback.py                   # 训练时推理回调函数
    ├── eval_utils.py                      # 评估mAP、CMC所需的util函数
    ├── meter.py
    ├── logging.py                         # 日志管理程序
    ├── lr_generator.py                    # 生成每个步骤的学习率
    ├── pcb.py                             # PCB模型结构、损失函数
    ├── rpp.py                             # PCB+RPP模型结构
    ├── resnet.py                          # resnet50模型结构
    ├── datasets                           # 包含处理Market-1501、DukeMTMC-reID、CUHK03数据集的程序
       ├── market.py                       # 处理Market-1501的程序
       ├── duke.py                         # 处理DukeMTMC-reID的程序
       ├── cuhk03.py                       # 处理CUHK03的程序
    ├── model_utils
       ├── config.py                       # 参数配置
       ├── device_adapter.py               # 设备配置
       ├── local_adapter.py                # 本地设备配置
       └── moxing_adapter.py               # modelarts设备配置
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
  └── export.py                            # 模型导出
  └── preprocess.py                        # 310推理预处理
  └── postprocess.py                       # 310推理后处理
```

## 脚本参数

- 配置PCB在Market-1501数据集评估。

```text
enable_modelarts: False                    # 是否开启modelarts云上训练作业训练
data_url: ""
train_url: ""
checkpoint_url: ""
enable_profiling: False
dataset_path: "/cache/dataset/"            # 数据集路径
output_path: "/cache/output/"              # 结果输出路径
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
log_save_path: "./log/PCB/market/eval"     # 日志保存路径
checkpoint_file_path: "/cache/load_checkpoint/PCB-60_202.ckpt"   # 断点加载路径

mindrecord_dir: "./MindRecord"             # MindRecord文件保存路径
dataset_name: "market"                     # 数据集简名
batch_size: 64                             # 一个数据批次大小
num_parallel_workers: 4

model_name: "PCB"                          # 模型名
use_G_feature: True                        # 评估时是否使用G feature，若不使用则代表使用H feature
```

- 配置PCB模型导出与推理。

```text
enable_modelarts: False                    # 是否开启modelarts云上训练作业训练
data_url: ""
train_url: ""
checkpoint_url: ""
enable_profiling: False
dataset_path: "/cache/dataset/"            # 数据集路径
output_path: "/cache/output/"              # 结果输出路径
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
checkpoint_file_path: "/cache/load_checkpoint/PCB-60_202.ckpt"   # 断点加载路径
batch_size: 1                              # 目前仅支持batch size为1的推理
model_name: "PCB"                          # 模型名
use_G_feature: True                        # 模型导出时是否使用G feature，若不使用则代表使用H feature，G/H feature选择的差异会影响导出模型的结构

device_id: 0
image_height: 384                          # 导出模型输入的高
image_width: 128                           # 导出模型输入的宽
file_name: "export_PCB_market_G"           # 导出的模型名
file_format: "MINDIR"                      # 导出的模型格式

preprocess_result_path: "./preprocess_Result"  #310推理预处理结果路径

query_prediction_path: "./query_result_files"  #query集合10推理结果输出路径
gallery_prediction_path: "./gallery_result_files"  #gallery集合310推理结果输出路径
```

## 训练过程(Ascend处理器环境运行)

# 用法：
```
bash run_standalone_train.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）
```
# 其中MODEL_NAME可从['PCB', 'RPP']中选择，DATASET_NAME['duke']。


# PCB在DukeMTMC-reID上训练
```
bash run_standalone_train.sh PCB duke ../../Datasets/DukeMTMC-reID ../config/train_PCB_duke.yaml ../../pretrained_resnet50.ckpt
```
# PCB+RPP在DukeMTMC-reID上训练（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）
```
bash run_standalone_train.sh RPP duke ../../Datasets/DukeMTMC-reID ../config/train_RPP_duke ../../pretrained_resnet50.ckpt
```
训练结果保存在脚本目录的output文件夹下，其中日志文件保存在./output/log/{MODEL_NAME}/{DATASET_NAME}/train下，断点文件保存在./output/checkpoint/{MODEL_NAME}/{DATASET_NAME}/train下，您可以在其中找到所需的信息。


- 使用DukeMTMC-reID数据集训练RPP

```log
# 单卡训练结果
epoch: 1 step: 258, loss is 23.096334
epoch time: 96244.296 ms, per step time: 373.040 ms
epoch: 2 step: 258, loss is 13.114418
epoch time: 33972.328 ms, per step time: 131.676 ms
epoch: 3 step: 258, loss is 8.97956
epoch time: 33965.507 ms, per step time: 131.649 ms
...
```


## 评估过程
```bash

# PCB在DukeMTMC-reID上使用G feature评估

bash run_eval.sh PCB duke ../../Datasets/DukeMTMC-reID ../config/eval_PCB_duke.yaml ./output/checkpoint/PCB/duke/train/PCB-60_258.ckpt True

# PCB+RPP在DukeMTMC-reID上使用G feature评估（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_eval.sh RPP duke ../../Datasets/DukeMTMC-reID ../config/eval_RPP_duke.yaml ./output/checkpoint/RPP/duke/train/RPP-40_258.ckpt True

评估结果保存在脚本目录的output/log/{MODEL_NAME}/{DATASET_NAME}/eval中。

### 结果

- RPP在DukeMTMC-reID数据集使用G feature进行评估

```log
Mean AP: 71.4%
CMC Scores        duke
  top-1          85.0%
  top-5          92.6%
  top-10         94.4%
```
## 推理过程

### 导出MindIR

导出mindir模型

```shell
python export.py --model_name [MODEL_NAME] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --checkpoint_file_path [CKPT_PATH] --use_G_feature [USE_G_FEATURE] --config_path [CONFIG_PATH]
```

参数model_name可从["PCB","RPP"]中选择。
参数file_name为导出的模型名称。
参数file_format 仅支持 "MINDIR"。
参数checkpoint_file_path为断点路径。
参数use_G_feature表示导出模型是否使用G feature，若不使用，则使用H feature。 feature类型的不同对应导出模型的结构也会不同。
参数config_path表示infer_310_config.yaml的路径

```shell
# 示例：
# 1、导出在Market-1501上训练后使用G feature的PCB模型。
python export.py --model_name "PCB" --file_name "PCB_duke_G" --file_format MINDIR --checkpoint_file_path ../PCB_duke.ckpt --use_G_feature True --config_path ./config/infer_310_config.yaml
```


### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [USE_G_FEATURE][CONFIG_PATH] [DEVICE_ID](optional)
```

- `DATASET_NAME` 选择范围：[market, duke, cuhk03]。
- `USE_G_FEATURE` 应与模型导出时的选项一致
- `CONFIG_PATH` 表示infer_310_config.yaml的路径
- `DEVICE_ID` 可选，默认值为0。

```bash
# 示例：
# 1、PCB在Market-1501上使用G feature进行推理。
bash run_infer_310.sh  ../../mindir/PCB_duke_G.mindir market ../../Datasets/DukeMTMC-reID True ../config/infer_310_config.yaml
```

### 结果

推理结果保存在脚本执行的当前路径，你可以在metrics.log中看到以下精度计算结果。
- PCB在DukeMTMC-reID数据集使用G feature进行推理

```log
Mean AP: 69.8%
  top-1          84.2%
  top-5          92.4%
  top-10         94.1%
```

由于PCB+RPP模型含有AvgPool3D算子，该算子在Ascend310环境暂不支持，因此这一部分未进行推理。

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。
