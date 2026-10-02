# Piper 机械臂 SDK使用说明

[EN](README.MD)

|![ubuntu18.04](https://img.shields.io/badge/Ubuntu-18.04-orange.svg)|![ubuntu20.04](https://img.shields.io/badge/Ubuntu-20.04-orange.svg)|![ubuntu22.04](https://img.shields.io/badge/Ubuntu-22.04-orange.svg)|
|---|---|---|
|![Pass](https://img.shields.io/badge/Pass-blue.svg)|![Pass](https://img.shields.io/badge/Pass-blue.svg)|![Pass](https://img.shields.io/badge/Pass-blue.svg)|

Test:

|![python3.6](https://img.shields.io/badge/Python-3.6-blue.svg)|![python3.8](https://img.shields.io/badge/Python-3.8-blue.svg)|![python3.10](https://img.shields.io/badge/Python-3.10-blue.svg)|
|---|---|---|
|![Pass](https://img.shields.io/badge/Pass-blue.svg)|![Pass](https://img.shields.io/badge/Pass-blue.svg)|![Pass](https://img.shields.io/badge/Pass-blue.svg)|

本SDK用于接收can数据帧，然后处理为自定义数据类型，没有包括数据偏移帧

|说明 |文档|
|---|---|
|接口函数的详细说明|[Interface_V2 README](./asserts/V2/INTERFACE_V2.MD)|
|机械臂DEMO|[`piper_sdk/demo/V2`](./piper_sdk/demo/V2/README.MD)|
|双臂的主从配置和读取数据|[double_piper](./asserts/double_piper.MD)|
|使用pyQT5的开源UI|[Piper_sdk_ui](<https://github.com/agilexrobotics/Piper_sdk_ui.git>)|
|CHANGELOG|[CHANGELOG](./CHANGELOG.MD)|
|Q&A|[Q&A](./asserts/Q&A.MD)|

<details open>
  <summary><h2 style="display:inline">1 软件安装方法</h2></summary>
  <div>

### 1.1 安装依赖

注: python-can版本应高于3.3.4

```shell
pip3 install python-can
```

### 1.2 安装 piper_sdk

注意是否在conda环境里

```bash
which pip3
```

这里提供三种方法，任选其一即可

#### 1.2.1 从pypi仓库安装

```shell
pip3 install piper_sdk
```

#### 1.2.2 拉取github仓库后安装

```shell
git clone https://github.com/agilexrobotics/piper_sdk.git
cd piper_sdk
pip3 install .
```

#### 1.2.3 从github的realease里下载whl文件安装

注意下面的X为你要的realease版本

```shell
pip3 install piper_sdk-X.X.X-py3-none-any.whl 
```

### 1.3 查看 piper_sdk 细节

```shell
pip3 show piper_sdk
```

0.0.x 为机械臂V1.5-2固件版本前的sdk支持

### 1.4 卸载 piper_sdk

```shell
pip3 uninstall piper_sdk
```

### 1.5 更新 piper_sdk

这里提供三种方法，任选其一即可

#### 1.5.1 从pypi仓库更新

可以使用`--upgrade`来更新

```shell
pip3 install --upgrade piper_sdk
```

可以先卸载旧版本然后重新安装，注意存储库有可能未同步，这里默认是官方源

```shell
pip3 uninstall piper_sdk
pip3 install piper_sdk
```

#### 1.5.2 从github仓库更新

```shell
git clone https://github.com/agilexrobotics/piper_sdk.git
cd piper_sdk
pip3 install .
```

#### 1.5.3 从github的realease里下载whl文件更新

注意下面的X为你要的realease版本

```shell
pip3 install piper_sdk-X.X.X-py3-none-any.whl 
```

</div>
</details>

<details open>
  <summary><h2 style="display:inline">2 使用can模块</h2></summary>
  <div>

注意此处的can模块仅支持机械臂自带的can模块，不支持其它can模块

安装can工具

```shell
sudo apt update && sudo apt install can-utils ethtool
```

这两个工具用于配置 CAN 模块

如果执行bash脚本出现`ip: command not found`，请安装ip指令，一般是`sudo apt-get install iproute2`

### 2.1 寻找can模块

执行

```bash
bash find_all_can_port.sh
```

输入密码后，如果can模块插入了电脑，并被电脑检测到，输出类似如下：

```bash
Both ethtool and can-utils are installed.
Interface can0 is connected to USB port 3-1.4:1.0
```

如果有多个，输出类似如下：

```bash
Both ethtool and can-utils are installed.
Interface can0 is connected to USB port 3-1.4:1.0
Interface can1 is connected to USB port 3-1.1:1.0
```

有多少个can模块就会有多少行类似`Interface can1 is connected to USB port 3-1.1:1.0`的输出

其中`can1`是系统找到的can模块名字，`3-1.1:1.0`是该can模块所链接的usb端口

如果之前已经激活过can模块并其名为其它名字，这里假设名字为`can_piper`则输出如下

```bash
Both ethtool and can-utils are installed.
Interface can_piper is connected to USB port 3-1.4:1.0
Interface can0 is connected to USB port 3-1.1:1.0
```

如果没有检测到can模块，则只会输出如下：

```bash
Both ethtool and can-utils are installed.
```

### 2.2 激活单个can模块, **此处使用`can_activate.sh`脚本**

激活单个can模块的情况分两种，一种是电脑只链接了一个can模块，一种是多个can模块插入电脑但是只激活其中一个。

#### 2.2.1 pc只插入一个usb转can模块

直接执行

```bash
bash can_activate.sh can0 1000000
```

此处`can0`可以改为任意名字，`1000000`为波特率，不可更改

#### 2.2.2 pc插入多个usb转can模块， 但每次只激活一个can模块

注： 此处用于同时使用机械臂和底盘的的情况

(1) 查看can模块插在usb端口的硬件地址。拔掉所有can模块，只将连接到机械臂的can模块插入PC，执行

```shell
bash find_all_can_port.sh
```

并记录下`USB port`的数值，例如`3-1.4:1.0`

(2) 激活can设备。假设上面的`USB port`数值为`3-1.4:1.0`，执行：

```bash
bash can_activate.sh can_piper 1000000 "3-1.4:1.0"
```

解释：**3-1.4:1.0硬件编码的usb端口插入的can设备，名字被重命名为can_piper，设定波特率为1000000，并激活**

(3) 检查是否激活成功

执行`ifconfig`查看是否有`can_piper`，如果有则can模块设置成功

### 2.3 同时激活多个can模块，**此处使用`can_muti_activate.sh`脚本**

首先确定有多少个官方can模块被插入到电脑，这里假设是2

注：**若当前电脑插入了5个can模块，可以只激活指定的can模块**

#### 2.3.1 记录每个can模块对应的usb口硬件地址

逐个拔插can模块并一一记录每个模块对应的usb口硬件地址

在`can_muti_activate.sh`中，`USB_PORTS`参数中元素的数量为预激活的can模块数量，现假设为2

(1) 然后can模块中的其中一个单独插入PC，执行

```shell
bash find_all_can_port.sh
```

并记录下`USB port`的数值，例如`3-1.4:1.0`

(2) 接着插入下一个can模块，注意**不可以**与上次can模块插入的usb口相同，然后执行

```shell
bash find_all_can_port.sh
```

记录下第二个can模块的`USB port`的数值，例如`3-1.1:1.0`

注：**如果未曾激活过，则第一个插入的can模块会默认是can0，第二个为can1，若激活过，名字为之前激活过的名称**

#### 2.3.2 预定义USB 端口、目标接口名称及其比特率

假设上面的操作记录的`USB port`数值分别为`3-1.4:1.0`、`3-1.1:1.0`，则将下面的`USB_PORTS["1-9:1.0"]="can_left:1000000"`的中括号内部的双引号内部的参数换为`3-1.4:1.0`和`3-1.1:1.0`.

最终结果为：

```bash
USB_PORTS["3-1.4:1.0"]="can_left:1000000"
USB_PORTS["3-1.1:1.0"]="can_right:1000000"
```

解释：**3-1.4:1.0硬件编码的usb端口插入的can设备，名字被重命名为can_left，波特率为1000000，并激活**

#### 2.3.3 激活多个can模块

执行`bash can_muti_activate.sh`

#### 2.3.4 查看多个can模块是否设置成功

执行`ifconfig`查看是不是有`can_left`和`can_right`

</div>
</details>

<details open>
  <summary><h2 style="display:inline">3 简单开始</h2></summary>
  <div>

从一个最简单的读取机械臂关节角开始

```python
import time
# 导入piper_sdk模块
from piper_sdk import *

if __name__ == "__main__":
    # 实例化interface，形参的默认参数如下
    # can_name(str -> default 'can0'): can port name
    # judge_flag(bool -> default True): 创建该实例时是否开启判断can模块，若使用的是非官方模块，请将其设置False
    # can_auto_init(bool): 创建该实例时是否自动进行初始化来打开can bus，如果设置为False，请在ConnectPort参数中将can_init形参设置为True
    # dh_is_offset([0,1] -> default 0x01): 使用的dh参数是新版dh还是旧版dh，S-V1.6-3以前的为旧版，S-V1.6-3固件及以后的为新版，对应fk的计算
    #             0 -> old
    #             1 -> new
    # start_sdk_joint_limit(bool -> default False):是否开启SDK的关节角度限位，会对反馈消息和控制消息都做限制
    # start_sdk_gripper_limit(bool -> default False):是否开启SDK的夹爪位置限位，会对反馈消息和控制消息都做限制
    # logger_level(LogLevel -> default LogLevel.WARNING):设定log的日志等级
    #         参数有如下可选: 
    #               LogLevel.DEBUG
    #               LogLevel.INFO
    #               LogLevel.WARNING
    #               LogLevel.ERROR
    #               LogLevel.CRITICAL
    #               LogLevel.SILENT
    # log_to_file(bool -> default False):是否打开log写入文件功能，True则打开，默认关闭
    # log_file_path(str -> default False):设定log写入文件的路径，默认在sdk路径下的log文件夹
    piper = C_PiperInterface(can_name="can0",
                            judge_flag=False,
                            can_auto_init=True,
                            dh_is_offset=1,
                            start_sdk_joint_limit=False,
                            start_sdk_gripper_limit=False,
                            logger_level=LogLevel.WARNING,
                            log_to_file=False,
                            log_file_path=None)
    # 开启can收发线程
    piper.ConnectPort()
    # 循环打印消息，注意所有的消息第一帧都是默认数值，比如关节角消息第一帧的消息内容默认为0
    while True:
        print(piper.GetArmJointMsgs())
        time.sleep(0.005)# 200hz
```

注意，读取关节反馈需要机械臂处在从臂模式下，如果为主臂模式，机械臂是不会反馈该消息的

您可以使用以下代码切换到从臂

```python
from piper_sdk import *
if __name__ == "__main__":
    piper = C_PiperInterface()
    piper.ConnectPort()
    piper.MasterSlaveConfig(0xFC, 0, 0, 0)
```

读取机械臂固件版本

```python
import time
from piper_sdk import *
if __name__ == "__main__":
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    time.sleep(0.03) # 需要时间去读取固件反馈帧，否则会反馈-0x4AF
    print(piper.GetPiperFirmwareVersion())
```

</div>
</details>

## 执行器辨识数据采集

采集入口为 `python -m piper_sdk.deployment.collect_actuator_data`，在 `sim` 环境运行。
每次运行选择一种模式（`move_j` / `mit`）和一种来源（`trajectory` / `policy`），
输出目录必须是新目录。**不带 `--preview` 会连接并操作机械臂**，正常结束后使用 MOVE_J 回零。

先离线生成轨迹，预览不连接 CAN：

```bash
conda activate sim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m piper_sdk.deployment.collect_actuator_data \
  --mode move_j --source trajectory --preview \
  --output_dir logs/actuator_data/preview
```

默认轨迹为三个姿态下的逐关节扫频和多关节组合运动，共约 21.59 分钟。
预览生成的 `trajectory_config.json` 包含姿态中心、幅度、频率和时长；
编辑后通过 `--trajectory_config <文件路径>` 使用，两种模式应使用同一配置。
保持基座静止、仅安装夹爪。指令周期为 20 ms，沿用每步 0.006 rad 的目标变化限制。
MOVE_J 速度比例固定为 5；MIT 的 Kp 为 `[4,4,4,4,4,2]`，
Kd 为 `[0.75,0.8,0.8,0.4,0.55,0.45]`，速度和力矩参考均为零。

分别采集两种模式的短段（前 1000 个轨迹点，包括移入、保持和 J1 扫频）：

```bash
python -m piper_sdk.deployment.collect_actuator_data \
  --mode move_j --source trajectory --max_steps 1000 \
  --output_dir logs/actuator_data/move_j_short
python -m piper_sdk.deployment.collect_actuator_data \
  --mode mit --source trajectory --max_steps 1000 \
  --output_dir logs/actuator_data/mit_short
```

完整采集时去掉 `--max_steps` 并使用新目录。两个模式的完整批次超过 30 分钟，
应在独立 tmux session 内运行。默认连接为 socketcand `192.168.123.162:29536`、
通道 `can_piper`，可使用 `--can_host`、`--can_port`、`--can_name` 修改。

策略补充数据使用显式指定的 checkpoint；`--policy_steps` 默认 500，
目标可通过 `--target_pos_b X Y Z` 或 `--random_target` 选择：

```bash
python -m piper_sdk.deployment.collect_actuator_data \
  --mode move_j --source policy \
  --checkpoint_path /mnt/nvme1/xinc/RoboDuet-IsaacLab/logs/rsl_rl/manipulation_direct/2026-09-05_22-21-22_history30/model_3999.pt \
  --target_pos_b 0.5 -0.2 0.4 --output_dir logs/actuator_data/move_j_policy
```

将 `--mode` 改为 `mit` 并更换输出目录即可采集另一模式。

当前扩充数据只使用 **MOVE_J**。批量入口默认生成 100 个不同的策略目标，
每个目标运行 1000 步（20 秒），复用上述单次采集器；每条单独初始化、记录并回零。
速度比例仍为 5，每步目标变化限制仍为 0.006 rad。

目标使用固定 seed 的 Sobol 采样覆盖现有策略训练范围：半径 0.4–0.7 m、
俯仰 0–1.2 rad、偏航 -1–1 rad，目标 `z >= 0`。这是策略命令坐标系，
对应夹爪在机械臂基座坐标系下的 z 减去 0.123 m；不代表运动途中所有连杆的离地高度。
显式位置目标沿用 `Manipulation` 的姿态计算方式，局部偏航为 0。

先生成目标清单和空间分布图，预览不需要 checkpoint，也不连接 CAN：

```bash
python -m piper_sdk.deployment.collect_move_j_batch \
  --num_trajectories 100 --policy_steps 1000 --seed 0 --preview \
  --output_dir logs/actuator_data/move_j_policy_100_preview
```

正式采集使用相同的数量和 seed，会生成相同的目标清单。策略运行本身共 33.3 分钟，
另加每条的初始化、模式接管、回零和文件处理时间。在独立 tmux session 中运行：

```bash
tmux new -s piper_move_j_100
conda activate sim
cd /mnt/nvme1/xinc/piper_sdk
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset -c 4-7,14-17 python -m piper_sdk.deployment.collect_move_j_batch \
  --num_trajectories 100 --policy_steps 1000 --seed 0 \
  --checkpoint_path /mnt/nvme1/xinc/RoboDuet-IsaacLab/logs/rsl_rl/manipulation_direct/2026-09-05_22-21-22_history30/model_3999.pt \
  --output_dir logs/actuator_data/move_j_policy_100
```

批次根目录保存 `targets.csv`、`targets.png` 和记录进度及失败原因的 `batch.json`；
各条数据保存在 `policy_000` 到 `policy_099`，对应运行日志位于根目录的 `policy_NNN.log`。
任一条失败立即结束批次。`Ctrl-C` 会转交给当前采集器，停止控制并保存已有记录。
输出目录必须是新目录，批量入口沿用 `--can_host`、`--can_port`、`--can_name`、`--device`。

默认 100 条提前按整条轨迹划为 80 条训练、10 条验证、10 条测试，分组保存在目标清单中。
后续训练时整条轨迹及其历史窗口必须属于同一组，不能把相邻采样点随机拆分。
100 条是首批规模，不是避免过拟合的保证；应结合已有逐关节扫频数据，
查看留出目标上的误差以及六个关节的位置、速度、电流覆盖，再决定补采范围。

100 条批量入口只运行策略，不会自动加入专用扫频或多关节正弦运动。
扩展的专用轨迹配置为 `piper_sdk/deployment/trajectory_configs/move_j_expanded.json`，
通过 `collect_actuator_data --source trajectory` 单独采集，不需要 checkpoint：

- 10 个关节姿态中心，覆盖更大的 J1–J6 角度范围；以最高 0.2 rad/s 移入，每个中心保持 5 秒。
- 每个中心对 J1–J6 分别执行 0.01、0.02 rad 幅度的扫频，各 20 秒，频率从 0.1 连续升到 2 Hz；段间保持 2 秒。
- 每个中心执行 40 秒六关节组合运动，幅度均为 0.04 rad，J1–J6 频率分别为 0.2、0.35、0.5、0.7、0.9、1.1 Hz。

完整扩展轨迹为 157018 个请求目标，共约 52.34 分钟，包含姿态间移入；
另加采集器初始化、模式接管和回零时间。
较高频段使用较小幅度，是为了让请求目标符合每步 0.006 rad 的变化限制。
正弦目标的最大变化速率为 `2*pi*f*A`，这些参数下单关节扫频不超过 0.252 rad/s，
组合运动各关节不超过 0.277 rad/s。原有三姿态、0.03/0.10 rad 幅度的低频数据仍保留。

离线生成完整请求序列：

```bash
python -m piper_sdk.deployment.collect_actuator_data \
  --mode move_j --source trajectory --preview \
  --trajectory_config piper_sdk/deployment/trajectory_configs/move_j_expanded.json \
  --output_dir logs/actuator_data/move_j_expanded_preview
```

100 条策略采集完成后，在 tmux 中单独运行扩展轨迹。两种采集不要同时控制机械臂：

```bash
taskset -c 4-7,14-17 python -m piper_sdk.deployment.collect_actuator_data \
  --mode move_j --source trajectory \
  --trajectory_config piper_sdk/deployment/trajectory_configs/move_j_expanded.json \
  --output_dir logs/actuator_data/move_j_expanded
```

这份关节轨迹的夹爪目标高度需通过正运动学换算；策略目标的 `z >= 0` 采样规则
不作用于 `--source trajectory`。本次已对完整扩展请求序列离线核对：命令坐标系下
夹爪目标 z 为 0.1032–0.5611 m，最大步间变化约 0.005525 rad。
本次核对结果保存在 `logs/actuator_data/move_j_expanded_preview/geometry_summary.json`；
该文件是本次额外进行 FK 计算的结果，普通 `--preview` 只生成轨迹与关节范围摘要。
这些数值描述请求轨迹，不是实机运动或碰撞检测结果。

| 输出 | 内容 |
| --- | --- |
| `metadata.json` | 模式、控制参数、连接与负载条件、固件、轨迹或策略信息、结束状态 |
| `can_frames.csv` | 每个原始收发帧及字节、本机单调时钟、CAN 来源时间戳、发送结果 |
| `steps.csv` | 请求目标、限幅后目标、从发送字节还原的目标角、反馈快照和阶段标记 |
| `trajectory.csv` / `trajectory_config.json` | 专用轨迹完整请求序列及配置 |
| `summary.json` | 实际周期、逐关节反馈频率、运动与电流范围、跟踪误差 |
| `response.png` | 按本机指令／观测时间绘制的位置、速度、电流概览 |

`initialization`、`mode_switch`、`transition`、`return`、`stop` 单独标记；
正式采集阶段为 `hold`、`sweep`、`multijoint`、`policy`。
两种模式的初始化和回位都使用 MOVE_J，不应混入 MIT 训练片段。
`steps.csv` 是异步反馈快照，原始时序以 `can_frames.csv` 为准。
CAN 来源时间戳可能来自另一台计算机，不能直接与本机时钟相减；
TX 本机时间表示发送调用，不表示电机实际接收时刻。
`motor_effort_estimate_nm` 是 SDK 的电流换算估计，尚非标定后的关节输出力矩。
发送或写入失败会结束采集并报告原因，已写入数据保留。

### MOVE_J 执行器 MLP 离线训练

在 `sim` 环境中，将已完成批次的原始 CAN 帧转换成 200 Hz 数据：

```bash
python -m piper_sdk.deployment.prepare_actuator_dataset \
  --input_dir logs/actuator_data/move_j_policy_100_manual \
  --output_dir logs/actuator_datasets/move_j_policy_100_200hz_8s --duration_s 8
```

`--input_dir` 可一次传入多个批次目录，例如合并原 100 条和新增 400 条：

```bash
python -m piper_sdk.deployment.prepare_actuator_dataset \
  --input_dir logs/actuator_data/move_j_policy_100_manual \
    logs/actuator_data/move_j_policy_200_10s_seed1_20261001_123957 \
  --output_dir logs/actuator_datasets/move_j_policy_500_200hz_8s --duration_s 8
```

各批原有的训练/验证/测试划分保持不变，上例合计 400/50/50 条；
轨迹名增加 `batch_000__` 等批次前缀，来源目录、原名和目标保存在清单中。
合并批次要求控制参数、策略 checkpoint、负载条件和固件一致。
后续两个训练入口均使用合并后的数据目录，LSTM 对比同一数据集新训练的 MLP。

只使用 `policy` 阶段，沿用目标清单中的整条轨迹分组。以本机单调时钟为准，
成功发送调用的结束时间对应目标更新，接收时间对应反馈更新；按过去最新帧
零阶保持到 5 ms 网格。各流共同覆盖区间之外的数据被排除，前 64 个历史点不计分。
`--duration_s` 默认 8 秒，从各流共同覆盖的对齐起点计算，包含 320 ms 历史/预热。
训练、验证、测试全部截取相同的 `[0,8s)` 前缀；每条轨迹保留 1,536 个计分点。
原始文件和之前生成的数据集不修改，`manifest.json` 记录截断长度和丢弃数量。
每条 `.npz` 保存 16 维历史输入、位置/速度/电流/估算力矩、运动标记，
以及各数值对应的接收/发送时间和 CAN 来源时间戳；`manifest.json` 保存处理数量。
新生成的数据还单独保存前 64 步的 `warmup_inputs`（位置误差、速度）及
`warmup_joint_pos_rad`（绝对位置）、`warmup_time_monotonic_ns`，供 LSTM 建立状态；
该部分没有力矩标签。

力矩标签由有符号电流（A）乘 SDK 系数，再应用固件版本修正。SDK 基础系数为
J1–J3 的 1.18125、J4–J6 的 0.95844；按[官方 Q&A 3.2](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/Q%26A.MD#32-sdk-to-obtain-motor-feedback-torque)，
固件 `<=1.8-2` 的 J1–J3 还需乘 4，J4–J6 不变。本批 `S-V1.7-3` 的最终系数为
`[4.725,4.725,4.725,0.95844,0.95844,0.95844]` N·m/A。
`manifest.json` 记录固件、基础系数、固件倍率及来源；训练入口核对系数与固件一致。
原始采集文件中的 SDK effort 保留原值，离线标签单独应用该修正。
标签仍是估算 N·m，并非独立标定的关节输出力矩。过去 100 ms 内实际位置或发送目标
的净变化速率超过 0.02 rad/s 即标为运动。训练使用普通 MSE，各计分样本权重相同，
不再对运动和保持进行类别平衡；分类仅用于分别报告误差。

六个关节独立训练并导出，训练应放在独立 tmux 会话：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m piper_sdk.deployment.train_actuator_net \
  --dataset_dir logs/actuator_datasets/move_j_policy_100_200hz_8s \
  --output_dir logs/actuator_models/move_j_mlp_8s --device cuda:0
```

两个入口的输出目录都需为新目录。训练默认 seed=0，各关节使用 seed+关节下标；
隐藏层为 64/64/32、ELU，AdamW 学习率 0.001、权重衰减 0.0001，batch size 2048。
最多训练 100 epoch，普通验证集 MSE 连续 15 epoch 不改善即停止。
可通过 `--epochs`、`--batch_size`、`--seed` 调整训练参数。

输入按 `pos_vel` 排列：位置误差历史在前，速度历史在后，索引为
`[0,1,2,4,8,16,32,64]`，对应过去 0–320 ms。归一化仅用训练集计算并封装进网络。
`models/j1.pt` 至 `j6.pt` 为 TorchScript，接收 `[batch,16]` 的 rad/rad/s 输入，
返回 `[batch,1]` 的估算 N·m。对应 `ActuatorNetMLPCfg` 的网络字段保存在
`actuator_network_config.json`，三个缩放参数均为 1；要求执行器调用周期为 5 ms。
物理力矩/速度/饱和限值和仿真接入需在后续回放验证时确定。

`metrics.json` 对比 MLP、按相同普通 MSE 拟合的常数与线性基线的整体/运动/保持误差；
`learning_curves.csv` 与 `plots/` 保存学习曲线及全部留出轨迹的估算力矩预测曲线。
`predictions/` 和 `trajectory_metrics.csv` 保存逐条预测与指标。测试集只在最佳模型
选定后评估，TorchScript 重载输出与原模型核对至 1e-5 N·m。

### MOVE_J 执行器 LSTM 离线训练

使用上述同一份 8 秒数据，并将同批次训练的 MLP 目录用于对比，输出目录需为新目录：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m piper_sdk.deployment.train_actuator_lstm \
  --dataset_dir logs/actuator_datasets/move_j_policy_100_200hz_8s \
  --mlp_dir logs/actuator_models/move_j_mlp_8s \
  --output_dir logs/actuator_models/move_j_lstm_8s --device cuda:0
```

在 `sim` 环境的独立 tmux 会话中运行。六关节分别使用两层、每层 32 单元的单向
LSTM 和线性输出层，每步只输入当前位置误差与速度。默认每批 8 条轨迹，按时间
顺序分成 256 步（1.28 秒）计算梯度；块间保留 `(h,c)` 的数值并截断梯度，
每条新轨迹从零状态开始。前 64 步用于预热，预热和末尾填充均不计损失或指标。

AdamW、学习率、权重衰减及普通 MSE 沿用 MLP，梯度范数裁剪至 1.0；
最多 100 epoch，普通验证集 MSE 连续 15 epoch 未改善则停止。`--batch_size`
表示轨迹条数，`--epochs` 和 `--seed` 可调整训练上限与种子；默认关节种子为 0–5。
训练使用 FP32，归一化仅由训练集正式计分样本计算并封装进模型。

训练前核对新数据与参考 MLP 的分组、计分输入、时间戳和标签完全相同；
原 MLP 数据目录须仍可访问。六个最佳 checkpoint 均选定后才进行测试集评估。
`dataset_acceptance.json` 记录样本和预热数量，`comparison.csv`、`metrics.json`
记录 LSTM、固定 MLP、原常数和线性基线的整体/运动/保持误差；`predictions/`、
`trajectory_metrics.csv` 和 `plots/` 保存全部留出轨迹的预测、指标与对比曲线。
裁剪后某条轨迹的单个关节若没有保持样本，该段指标记为缺失，样本数记为 0。
重训前后的比较必须在相同的 8 秒时间戳上重新计算指标，不直接比较旧的全时段指标。

`models/j1.pt` 至 `j6.pt` 遵循 Isaac Lab `ActuatorNetLSTM`：
`forward(x, (h,c))` 返回 `(torque_est_nm, (h_next,c_next))`。
单步 `x` 为 `[batch,1,2]`，输出为 `[batch,1,1]`，状态为 `[2,batch,32]`；
保留 `.lstm` 子模块供 Isaac Lab 查询结构。调用周期为 5 ms，每个环境重置时清空
两种状态，步间连续传递；无需 MLP 的历史索引和外部缩放字段。

导出时在完整留出轨迹上核对脚本/非脚本单步输出、状态、分块推理及清零后回放。
`actuator_network_config.json` 只提供网络参数；估算力矩标定、物理限值与仿真接入
仍需后续完成。训练或导出失败直接终止，`metrics.json` 保留失败原因。

### MOVE_J 全臂联合 LSTM

联合模型每步读取 `[error_J1..J6, position_J1..J6, velocity_J1..J6]` 共 18 维，
其中误差为实际发送目标减实际角度；输出六个关节的估算力矩。每条机械臂轨迹
维护一份状态。默认网络为两层、每层 64 单元的单向 LSTM，加六维线性输出层。
通过 `--num_layers` 和 `--hidden_size` 指定规模：加宽使用
`--num_layers 2 --hidden_size 128`，加深使用 `--num_layers 3 --hidden_size 64`。
循环状态按模型的实际层数和隐藏宽度创建，checkpoint 和配置记录相应结构。

先用更新后的数据处理入口生成新目录，补齐原始 CAN 对齐得到的预热段绝对角度。
原有处理数据和模型保留，计分时刻、标签、原始分组保持一致：

```bash
python -m piper_sdk.deployment.prepare_actuator_dataset \
  --input_dir logs/actuator_data/move_j_policy_100_manual \
    logs/actuator_data/move_j_policy_200_10s_seed1_20261001_123957 \
  --output_dir logs/actuator_datasets/move_j_policy_500_200hz_8s_joint --duration_s 8

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m piper_sdk.deployment.train_actuator_joint_lstm \
  --dataset_dir logs/actuator_datasets/move_j_policy_500_200hz_8s_joint \
  --mlp_dir logs/actuator_models/20261001_161756_500_firmware_torque_corrected/move_j_mlp \
  --lstm_dir logs/actuator_models/20261001_161756_500_firmware_torque_corrected/move_j_lstm \
  --output_dir logs/actuator_models/move_j_joint_lstm_8s --device cuda:0
```

在 `sim` 环境的独立 tmux 会话中运行。默认 seed=0，每批 8 条轨迹，分块长度
256 步；轨迹间清零状态、块间保留状态并截断梯度。前 64 步预热和填充不计损失。
训练、验证、测试均使用前 8 秒，计分区间为 0.32–8 秒。

18 个输入和六个输出分别使用训练集计分样本的均值、标准差归一化。损失是
归一化后六个输出的普通 MSE，不使用运动/保持类别权重。按六个输出平均的
验证归一化 MSE 选择一个共同 checkpoint；最多 100 epoch，15 epoch 未改善
则停止。AdamW 学习率 0.001、权重衰减 0.0001，梯度范数上限 1.0，使用 FP32。
对比固定训练轮数时使用 `--epochs 100 --early_stopping_patience 0` 或
`--epochs 200 --early_stopping_patience 0`，两次分别从相同 seed 初始化并使用独立输出目录。
此时训练跑满指定轮数，仍按验证指标保留最佳模型；最后一轮权重另存为
`checkpoints/joint_lstm_last.pt`，避免将最佳 epoch 与训练总轮数混淆。

`checkpoints/joint_lstm_best.pt` 和 `models/joint_lstm.pt` 保存最佳模型及 TorchScript，
归一化封装在网络内。`forward(x, (h,c))` 接收 `[batch,time,18]`，返回
`[batch,time,6]` 的估算 N·m 及 `[num_layers,batch,hidden_size]` 的两种状态；数据周期为 5 ms。
`model_config.json` 保存输入顺序、单位和结构，模型作为独立产物交付。
联合模型的导出数值核对统一使用 CUDA 原生循环内核，在核对期间关闭 cuDNN，
使长序列、单步和 TorchScript 使用相同计算路径；训练仍使用 cuDNN。
输出、状态传递及清零回放的 `1e-5` 验收阈值保持不变。

训练完成后，在相同的验证/测试轨迹上比较联合 LSTM、固定的独立 MLP、独立 LSTM、
使用当前 18 维全臂输入的线性基线和常数基线。两种新增基线仅由训练集拟合。
独立模型直接加载已有 TorchScript 并在当前留出轨迹上重新推理，不要求它们与
新联合模型使用同一训练批次；控制、预处理和力矩估算口径必须一致。
`dataset_acceptance.json` 记录参考模型的原训练数据规模，比较时需区分数据增量与模型结构的影响。
`metrics.json`、`comparison.csv` 分别报告各关节整体/运动/保持的 MAE、RMSE、R²，
并按来源批次拆分结果；`predictions/`、`plots/` 保存全部留出轨迹的预测和曲线。
本实验同时改变输入信息、联合输出、共享状态和模型容量，收益不能单独归因于
“联合训练”。全部标签仍为 SDK 电流换算估计。

2026-10-01 的固件单位修正产物保存在
`logs/actuator_models/20261001_161756_500_firmware_torque_corrected/`，
覆盖最新 500 条实验的独立 MLP、独立 LSTM 和联合 LSTM。
此前历史产物遗漏 J1–J3 的固件倍率，其力矩值、MAE、RMSE 均需乘 4。
修正版直接将模型输出反归一化的均值和标准差乘以对应倍率，学习到的权重不变，
不重新优化模型；R²、归一化损失与各关节相对改善比例不变。修正后的模型已内置
倍率，使用时不要再乘 4。`correction_report.json` 保存数据及推理数值核对结果。

## 注意事项

- 需要先激活can设备，并且设置正确的波特率，才可以读取机械臂消息或者控制机械臂
- C_PiperInterface 接口类在实例化时可传入激活的can路由名称，这个名称可以通过`ifconfig`得到
- 执行can发送，终端反馈`SendCanMessage(SEND_MESSAGE_FAILED (100017))`，是can模块没有成功连接设备，先检查模块与机械臂的连接状态，然后将机械臂断电后上电，再尝试发送
- sdk的interface在创建实例后会检测是否激活自带的can模块，如果是其它can设备，可以将第二个形参设定为False，如：`piper = C_PiperInterface_V2("can0",False)`
- **机械臂的mit协议控制单关节电机为高级功能,使用时需注意,此协议的使用不当会导致机械臂损坏！！！**

## 联系我们

您可以在github上提issue

也可以加入discord<https://discord.gg/wrKYTxwDBd>
