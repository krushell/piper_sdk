import math
import time

from piper_sdk import C_PiperInterface_V2

# 默认CAN设备名称
CAN_NAME = "can0"
TARGET_RAD = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)
SPEED_PERCENT = 5
TARGET_TOLERANCE_DEG = 0.5
FEEDBACK_TIMEOUT_SECONDS = 0.5
MOTION_TIMEOUT_SECONDS = 20.0

# 初始化 Piper CAN 总线接口
piper = C_PiperInterface_V2(CAN_NAME)
control_started = False


def quick_stop():
    """Best-effort software stop. A physical emergency stop is still required."""
    stop_errors = []
    for _ in range(5):
        try:
            piper.MotionCtrl_1(0x01, 0, 0)
        except Exception as exc:
            stop_errors.append(exc)
        time.sleep(0.01)

    if stop_errors:
        print(f"警告：快速急停指令发送失败：{stop_errors[-1]}")


try:
    # 启动CAN的通信端口 用于发送指令和读取关节状态
    piper.ConnectPort()

    # 等待3s 关节和机械臂状态反馈，避免读取 SDK 的初始零值。
    feedback_deadline = time.monotonic() + 3.0
    while True:
        joint_msg = piper.GetArmJointMsgs()
        status_msg = piper.GetArmStatus()
        # 收到piper的反馈 可以退出等待
        if joint_msg.Hz > 0 and status_msg.Hz > 0:
            break
        if time.monotonic() > feedback_deadline:
            raise RuntimeError("没有收到完整反馈，请检查 CAN、机械臂电源和从臂模式")
        time.sleep(0.02)
    
    target_deg = [math.degrees(angle) for angle in TARGET_RAD]

    for index, (angle, limits) in enumerate(
        zip(target_deg, JOINT_LIMITS_DEG), start=1
    ):
        lower, upper = limits
        if not lower <= angle <= upper:
            raise ValueError(
                f"J{index} 目标角度越界：{angle:.3f}°，"
                f"允许范围 [{lower:.1f}, {upper:.1f}]°"
            )

    print(f"目标位置(rad)：{[round(value, 4) for value in TARGET_RAD]}")
    print(f"目标位置(deg)：{[round(value, 3) for value in target_deg]}")

    # 单位为0.001角度 
    joint_msg = piper.GetArmJointMsgs()
    status_msg = piper.GetArmStatus()
    init_joint_state = joint_msg.joint_state
    init_joint_deg = [
        init_joint_state.joint_1 / 1000.0,
        init_joint_state.joint_2 / 1000.0,
        init_joint_state.joint_3 / 1000.0,
        init_joint_state.joint_4 / 1000.0,
        init_joint_state.joint_5 / 1000.0,
        init_joint_state.joint_6 / 1000.0,
    ]

    if joint_msg.Hz <= 0 or status_msg.Hz <= 0:
        raise RuntimeError("CAN 反馈频率为零")

    arm_status = status_msg.arm_status.arm_status # 0x00字段表示状态正常
    if int(arm_status) != 0:
        raise RuntimeError(f"机械臂启动前状态异常：{arm_status}")
    # 绝对关节位置
    target = [round(angle * 1000) for angle in target_deg]

    control_started = True
    enable_deadline = time.monotonic() + 5.0
    # 使能机械臂
    while not piper.EnablePiper():
        if time.monotonic() > enable_deadline:
            raise RuntimeError("机械臂使能失败")
        time.sleep(0.01)

    motion_deadline = time.monotonic() + MOTION_TIMEOUT_SECONDS
    # 记录最后一次收到关节反馈的时间戳
    last_feedback_stamp = joint_msg.time_stamp
    last_feedback_time = time.monotonic()

    while True:
        piper.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
        piper.JointCtrl(*target)

        joint_msg = piper.GetArmJointMsgs()
        status_msg = piper.GetArmStatus()
        now = time.monotonic()

        if joint_msg.time_stamp != last_feedback_stamp:
            last_feedback_stamp = joint_msg.time_stamp
            last_feedback_time = now
        elif now - last_feedback_time > FEEDBACK_TIMEOUT_SECONDS:
            raise RuntimeError("关节反馈中断")

        if joint_msg.Hz <= 0 or status_msg.Hz <= 0:
            raise RuntimeError("CAN 反馈频率为零")

        arm_status = status_msg.arm_status.arm_status
        if int(arm_status) != 0:
            raise RuntimeError(f"机械臂状态异常：{arm_status}")

        state = joint_msg.joint_state
        current_joint_deg = [
            state.joint_1 / 1000.0,
            state.joint_2 / 1000.0,
            state.joint_3 / 1000.0,
            state.joint_4 / 1000.0,
            state.joint_5 / 1000.0,
            state.joint_6 / 1000.0,
        ]
        current_joint_rad = [math.radians(angle) for angle in current_joint_deg]
        errors_deg = [
            abs(actual - target)
            for actual, target in zip(current_joint_deg, target_deg)
        ]
        max_error_deg = max(errors_deg)
        print(
            f"\r当前(rad)={[round(value, 3) for value in current_joint_rad]}，"
            f"最大误差={max_error_deg:.3f}°",
            end="",
            flush=True,
        )

        if max_error_deg <= TARGET_TOLERANCE_DEG:
            print("\n全部关节已到达目标位置并保持使能。")
            break

        if now > motion_deadline:
            raise TimeoutError(
                f"关节在 {MOTION_TIMEOUT_SECONDS:.0f} 秒内没有全部到达目标位置"
            )

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n收到 Ctrl+C。")
    if control_started:
        print("正在发送快速急停...")
        quick_stop()
        print("已完成软件急停。")

except Exception:
    if control_started:
        print("\n程序异常，正在发送快速急停...")
        quick_stop()
    raise

finally:
    piper.DisconnectPort()
