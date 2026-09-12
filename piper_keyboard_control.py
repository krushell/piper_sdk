#!/usr/bin/env python3
import argparse
import os
import select
import sys
import termios
import time
import tty

from piper_sdk import C_PiperInterface_V2
from piper_sdk.joint_feedback_ipc import (
    DEFAULT_JOINT_FEEDBACK_ENDPOINT,
    PiperJointFeedbackPublisher,
)


CAN_NAME = "can_piper"
CAN_HOST = "192.168.123.162"
CAN_PORT = 29536
STEP_DEG = 10.0
SPEED_PERCENT = 2
MOVE_J_MIT_MODE = 0xAD
TARGET_TOLERANCE_DEG = 1
FEEDBACK_TIMEOUT_SECONDS = 0.5
MOTION_TIMEOUT_SECONDS = 20.0
ARM_RECOVERY_TIMEOUT_SECONDS = 5.0
ARM_RECOVERY_STABLE_SAMPLES = 10
RESET_SETTLE_SECONDS = 1.0
ENABLE_TIMEOUT_SECONDS = 8.0
ENABLE_COMMAND_INTERVAL_SECONDS = 1.0
ENABLE_POST_COMMAND_SETTLE_SECONDS = 0.5
ENABLE_STABLE_SAMPLES = 10
REQUIRED_ENABLE_COMMANDS = 2
CONTROL_READY_TIMEOUT_SECONDS = 5.0
CONTROL_READY_STABLE_SAMPLES = 20
KEY_RELEASE_GAP_SECONDS = 0.15
LOOP_PERIOD_SECONDS = 0.02
FEEDBACK_LIMIT_MARGIN_DEG = 5.0
PIPER_CPU_AFFINITY = "4-5,14-15"

JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)

KEY_BINDINGS = {
    "q": (0, 1),
    "a": (0, -1),
    "w": (1, 1),
    "s": (1, -1),
    "e": (2, 1),
    "d": (2, -1),
    "r": (3, 1),
    "f": (3, -1),
    "t": (4, 1),
    "g": (4, -1),
    "y": (5, 1),
    "h": (5, -1),
}


def joint_degrees(joint_msg):
    state = joint_msg.joint_state
    return [
        state.joint_1 / 1000.0,
        state.joint_2 / 1000.0,
        state.joint_3 / 1000.0,
        state.joint_4 / 1000.0,
        state.joint_5 / 1000.0,
        state.joint_6 / 1000.0,
    ]


def parse_cpu_affinity(value):
    cpus = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if last < first:
                raise ValueError(f"无效 CPU 范围：{part}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU affinity 不能为空")
    return cpus


def configure_cpu_affinity(value):
    requested = parse_cpu_affinity(value)
    available = os.sched_getaffinity(0)
    unavailable = requested - available
    if unavailable:
        raise ValueError(f"CPU 不可用：{sorted(unavailable)}")
    os.sched_setaffinity(0, requested)
    return tuple(sorted(requested))


def sdk_target(target_deg):
    return [round(angle * 1000) for angle in target_deg]


def quick_stop(piper):
    """Best-effort software stop; it does not replace the physical E-stop."""
    stop_errors = []
    for _ in range(5):
        try:
            piper.MotionCtrl_1(0x01, 0, 0)
        except Exception as exc:
            stop_errors.append(exc)
        time.sleep(0.01)

    if stop_errors:
        print(f"警告：快速急停指令发送失败：{stop_errors[-1]}")


def arm_status_diagnostics(status_msg):
    status = status_msg.arm_status
    return (
        f"status={status.arm_status}, ctrl_mode={status.ctrl_mode}, "
        f"mode_feed={status.mode_feed}, "
        f"err_code=0x{int(status.err_code) & 0xFFFF:04X}"
    )


def wait_for_arm_recovery(piper, reset_sent_at):
    deadline = time.monotonic() + ARM_RECOVERY_TIMEOUT_SECONDS
    settle_deadline = reset_sent_at + RESET_SETTLE_SECONDS
    stable_samples = 0
    last_status_stamp = None
    last_joint_stamp = piper.GetArmJointMsgs().time_stamp
    last_low_spd_stamp = piper.GetArmLowSpdInfoMsgs().time_stamp
    fresh_joint_feedback = False
    fresh_low_spd_feedback = False
    last_diagnostics = "尚未收到 reset 后的新状态帧"

    while time.monotonic() <= deadline:
        status_msg = piper.GetArmStatus()
        joint_msg = piper.GetArmJointMsgs()
        low_spd_msg = piper.GetArmLowSpdInfoMsgs()
        if (
            status_msg.Hz <= 0
            or joint_msg.Hz <= 0
            or low_spd_msg.Hz <= 0
        ):
            stable_samples = 0
            last_diagnostics = "状态、关节或六轴驱动器低速反馈频率为零"
            time.sleep(0.01)
            continue

        if joint_msg.time_stamp != last_joint_stamp:
            last_joint_stamp = joint_msg.time_stamp
            fresh_joint_feedback = True
        if low_spd_msg.time_stamp != last_low_spd_stamp:
            last_low_spd_stamp = low_spd_msg.time_stamp
            fresh_low_spd_feedback = True
        if status_msg.time_stamp == last_status_stamp:
            time.sleep(0.005)
            continue
        last_status_stamp = status_msg.time_stamp

        arm_status = int(status_msg.arm_status.arm_status)
        err_code = int(status_msg.arm_status.err_code)
        last_diagnostics = (
            f"{arm_status_diagnostics(status_msg)}, "
            f"driver_hz={low_spd_msg.Hz:.1f}, "
            f"enable_status={piper.GetArmEnableStatus()}"
        )
        recovered = (
            time.monotonic() >= settle_deadline
            and arm_status == 0
            and err_code == 0
            and fresh_joint_feedback
            and fresh_low_spd_feedback
        )
        if recovered:
            stable_samples += 1
            if stable_samples >= ARM_RECOVERY_STABLE_SAMPLES:
                return
        else:
            stable_samples = 0
            if arm_status not in (0x00, 0x01, 0x05):
                raise RuntimeError(
                    f"reset 后出现不可恢复状态：{last_diagnostics}"
                )

        time.sleep(0.01)

    raise RuntimeError(
        "等待 reset 后机械臂恢复超时："
        f"{last_diagnostics}, "
        f"fresh_joint_feedback={fresh_joint_feedback}, "
        f"fresh_driver_feedback={fresh_low_spd_feedback}"
    )


def enable_arm_and_wait(piper):
    deadline = time.monotonic() + ENABLE_TIMEOUT_SECONDS
    enable_command_count = 0
    last_enable_command_time = None
    last_status_stamp = None
    last_low_spd_stamp = piper.GetArmLowSpdInfoMsgs().time_stamp
    fresh_driver_feedback = False
    stable_samples = 0
    last_diagnostics = "尚未收到使能后的新反馈"

    while time.monotonic() <= deadline:
        now = time.monotonic()
        enable_status = piper.GetArmEnableStatus()
        need_enable_command = (
            enable_command_count < REQUIRED_ENABLE_COMMANDS
            or not all(enable_status)
        )
        command_interval_elapsed = (
            last_enable_command_time is None
            or now - last_enable_command_time
            >= ENABLE_COMMAND_INTERVAL_SECONDS
        )
        if need_enable_command and command_interval_elapsed:
            piper.EnableArm(7)
            enable_command_count += 1
            last_enable_command_time = now
            last_low_spd_stamp = piper.GetArmLowSpdInfoMsgs().time_stamp
            fresh_driver_feedback = False

        status_msg = piper.GetArmStatus()
        low_spd_msg = piper.GetArmLowSpdInfoMsgs()
        if low_spd_msg.time_stamp != last_low_spd_stamp:
            last_low_spd_stamp = low_spd_msg.time_stamp
            fresh_driver_feedback = True

        arm_status = int(status_msg.arm_status.arm_status)
        err_code = int(status_msg.arm_status.err_code)
        enable_status = piper.GetArmEnableStatus()
        last_diagnostics = (
            f"{arm_status_diagnostics(status_msg)}, "
            f"driver_hz={low_spd_msg.Hz:.1f}, "
            f"enable_status={enable_status}, "
            f"enable_commands={enable_command_count}"
        )

        new_status_frame = status_msg.time_stamp != last_status_stamp
        if new_status_frame:
            last_status_stamp = status_msg.time_stamp

        enabled = (
            enable_command_count >= REQUIRED_ENABLE_COMMANDS
            and last_enable_command_time is not None
            and now - last_enable_command_time
            >= ENABLE_POST_COMMAND_SETTLE_SECONDS
            and fresh_driver_feedback
            and low_spd_msg.Hz > 0
            and all(enable_status)
            and arm_status == 0
            and err_code == 0
        )
        if enabled and new_status_frame:
            stable_samples += 1
            if stable_samples >= ENABLE_STABLE_SAMPLES:
                return
        elif not enabled:
            stable_samples = 0
            if arm_status not in (0x00, 0x05):
                raise RuntimeError(
                    f"重新使能期间机械臂状态异常：{last_diagnostics}"
                )

        time.sleep(0.01)

    raise RuntimeError(
        "重新使能机械臂超时："
        f"{last_diagnostics}, "
        f"fresh_driver_feedback={fresh_driver_feedback}"
    )


def wait_for_move_j_control(piper):
    """Enter MOVE_J while tolerating only its known 0x05 transition state."""
    deadline = time.monotonic() + CONTROL_READY_TIMEOUT_SECONDS
    stable_samples = 0
    last_status_stamp = piper.GetArmStatus().time_stamp
    last_diagnostics = "尚未收到模式切换后的状态帧"

    while time.monotonic() <= deadline:
        # 与 manipulation 一致：模式稳定前只发送 0x151，不发送 JointCtrl。
        piper.MotionCtrl_2(
            0x01,
            0x01,
            SPEED_PERCENT,
            MOVE_J_MIT_MODE,
        )
        time.sleep(LOOP_PERIOD_SECONDS)

        status_msg = piper.GetArmStatus()
        joint_msg = piper.GetArmJointMsgs()
        low_spd_msg = piper.GetArmLowSpdInfoMsgs()
        if (
            status_msg.Hz <= 0
            or joint_msg.Hz <= 0
            or low_spd_msg.Hz <= 0
        ):
            stable_samples = 0
            last_diagnostics = "状态、关节或六轴驱动器低速反馈频率为零"
            continue

        if status_msg.time_stamp == last_status_stamp:
            continue
        last_status_stamp = status_msg.time_stamp

        status = status_msg.arm_status
        arm_status = int(status.arm_status)
        ready = (
            arm_status == 0
            and int(status.ctrl_mode) == 0x01
            and int(status.mode_feed) == 0x01
            and int(status.err_code) == 0
            and all(piper.GetArmEnableStatus())
        )
        last_diagnostics = arm_status_diagnostics(status_msg)

        if ready:
            stable_samples += 1
            if stable_samples >= CONTROL_READY_STABLE_SAMPLES:
                return
        else:
            stable_samples = 0
            if arm_status not in (0x00, 0x05):
                raise RuntimeError(
                    f"准备 MOVE_J 控制时机械臂状态异常：{last_diagnostics}"
                )

    raise RuntimeError(
        "等待 MOVE_J 控制模式稳定超时："
        f"{last_diagnostics}"
    )


def prepare_arm_control(piper):
    status_msg = piper.GetArmStatus()
    joint_msg = piper.GetArmJointMsgs()
    low_spd_msg = piper.GetArmLowSpdInfoMsgs()
    if status_msg.Hz <= 0 or joint_msg.Hz <= 0 or low_spd_msg.Hz <= 0:
        raise RuntimeError("准备控制前 CAN 反馈频率为零")

    status = status_msg.arm_status
    arm_status = int(status.arm_status)
    err_code = int(status.err_code)
    ctrl_mode = int(status.ctrl_mode)
    if arm_status not in (0x00, 0x01, 0x05):
        raise RuntimeError(
            "机械臂存在不可自动恢复故障："
            f"{arm_status_diagnostics(status_msg)}"
        )

    can_control_is_healthy = (
        arm_status == 0
        and err_code == 0
        and ctrl_mode == 0x01
    )
    reset_performed = False
    if can_control_is_healthy:
        print("机械臂处于正常 CAN_CTRL，跳过 reset")
    else:
        reset_sent_at = time.monotonic()
        piper.MotionCtrl_1(0x02, 0, 0)
        wait_for_arm_recovery(piper, reset_sent_at)
        reset_performed = True
        print("机械臂 reset 恢复完成")

    if reset_performed or not all(piper.GetArmEnableStatus()):
        enable_arm_and_wait(piper)
        print("机械臂重新使能完成")

    wait_for_move_j_control(piper)
    print("机械臂控制准备完成（MOVE_J 0xAD）")


def hold_current_position(piper, joint_msg):
    target = sdk_target(joint_degrees(joint_msg))
    for _ in range(10):
        piper.MotionCtrl_2(
            0x01,
            0x01,
            SPEED_PERCENT,
            MOVE_J_MIT_MODE,
        )
        piper.JointCtrl(*target)
        time.sleep(LOOP_PERIOD_SECONDS)


def build_control_limits(pose_deg, label, show_warnings=True):
    control_limits = []
    for index, (angle, limits) in enumerate(
        zip(pose_deg, JOINT_LIMITS_DEG), start=1
    ):
        lower, upper = limits
        if not (
            lower - FEEDBACK_LIMIT_MARGIN_DEG
            <= angle
            <= upper + FEEDBACK_LIMIT_MARGIN_DEG
        ):
            raise RuntimeError(
                f"{label} J{index}={angle:.3f}° 超出范围 "
                f"[{lower:.1f}, {upper:.1f}]°，且超过允许的 "
                f"{FEEDBACK_LIMIT_MARGIN_DEG:.1f}° 零位偏差"
            )

        if not lower <= angle <= upper and show_warnings:
            print(
                f"提示：J{index} 当前反馈 {angle:.3f}° 略超文档范围 "
                f"[{lower:.1f}, {upper:.1f}]°；本次会话允许保持或返回该位置，"
                "但不允许继续向外越界。"
            )

        control_limits.append((min(lower, angle), max(upper, angle)))

    return tuple(control_limits)


def print_controls():
    print("\n键盘控制（每次 10°，到位后才接受下一次按键）：")
    print("  J1: q 增加 / a 减少")
    print("  J2: w 增加 / s 减少")
    print("  J3: e 增加 / d 减少")
    print("  J4: r 增加 / f 减少")
    print("  J5: t 增加 / g 减少")
    print("  J6: y 增加 / h 减少")
    print("  x : 保持当前位置并退出")
    print("  空格或 Ctrl+C: 软件快速急停并退出")
    print("请勿长按按键；实体急停必须保持触手可及。\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--can_name",
        default=CAN_NAME,
        help="PC2 上由 socketcand 暴露的 CAN 设备名称",
    )
    parser.add_argument(
        "--can_host",
        default=CAN_HOST,
        help="PC2 socketcand 地址",
    )
    parser.add_argument(
        "--can_port",
        type=int,
        default=CAN_PORT,
        help="PC2 socketcand TCP 端口",
    )
    parser.add_argument(
        "--joint-feedback-endpoint",
        default=DEFAULT_JOINT_FEEDBACK_ENDPOINT,
        help="向按钮观测器发布关节反馈的本机 Unix 数据报端点",
    )
    parser.add_argument(
        "--cpu-affinity",
        default=PIPER_CPU_AFFINITY,
        help="PiPER 控制进程独占使用的 CPU；默认避开 A2 和视觉进程",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    control_cpus = configure_cpu_affinity(args.cpu_affinity)
    if not sys.stdin.isatty():
        raise RuntimeError("该脚本必须在交互式终端中运行")

    feedback_publisher = PiperJointFeedbackPublisher(
        args.joint_feedback_endpoint
    )
    print(
        f"PiPER 控制 CPU={list(control_cpus)}；"
        f"关节反馈 IPC={args.joint_feedback_endpoint}"
    )

    piper = C_PiperInterface_V2(
        args.can_name,
        judge_flag=False,
        can_auto_init=False,
    )
    piper.CreateCanBus(
        can_name=args.can_name,
        bustype="socketcand",
        judge_flag=False,
        host=args.can_host,
        port=args.can_port,
        tcp_tune=True,
    )
    control_started = False
    emergency_stopped = False
    old_terminal_settings = None

    try:
        piper.ConnectPort()

        feedback_deadline = time.monotonic() + 3.0
        while True:
            joint_msg = piper.GetArmJointMsgs()
            status_msg = piper.GetArmStatus()
            low_spd_msg = piper.GetArmLowSpdInfoMsgs()
            if (
                joint_msg.Hz > 0
                and status_msg.Hz > 0
                and low_spd_msg.Hz > 0
            ):
                break
            if time.monotonic() > feedback_deadline:
                raise RuntimeError(
                    "没有收到完整反馈，请检查 CAN、机械臂电源和从臂模式"
                )
            time.sleep(LOOP_PERIOD_SECONDS)

        initial_deg = joint_degrees(joint_msg)
        build_control_limits(initial_deg, "当前位置")

        arm_status = int(status_msg.arm_status.arm_status)
        if arm_status not in (0x00, 0x01, 0x05):
            raise RuntimeError(
                "机械臂启动前存在不可自动恢复故障："
                f"{arm_status_diagnostics(status_msg)}"
            )

        print(f"当前关节角度：{[round(value, 3) for value in initial_deg]}")
        print(f"单次步长：{STEP_DEG:.1f}°，速度：{SPEED_PERCENT}%")
        confirmation = input(
            "确认工作空间安全且实体急停可用后，输入 KEYBOARD 开始："
        )
        if confirmation.strip() != "KEYBOARD":
            raise SystemExit("已取消，未使能或发送运动指令。")

        confirmed_joint_msg = piper.GetArmJointMsgs()
        confirmed_status_msg = piper.GetArmStatus()
        confirmed_deg = joint_degrees(confirmed_joint_msg)
        max_pose_change = max(
            abs(confirmed - original)
            for confirmed, original in zip(confirmed_deg, initial_deg)
        )
        if confirmed_joint_msg.Hz <= 0 or confirmed_status_msg.Hz <= 0:
            raise RuntimeError("确认时 CAN 反馈频率为零")
        if max_pose_change > TARGET_TOLERANCE_DEG:
            raise RuntimeError("确认期间关节姿态发生变化，请重新运行脚本")
        confirmed_arm_status = int(
            confirmed_status_msg.arm_status.arm_status
        )
        if confirmed_arm_status not in (0x00, 0x01, 0x05):
            raise RuntimeError(
                "确认时机械臂存在不可自动恢复故障："
                f"{arm_status_diagnostics(confirmed_status_msg)}"
            )

        build_control_limits(confirmed_deg, "确认位置", show_warnings=False)
        control_started = True
        prepare_arm_control(piper)

        # 以使能且 MOVE_J 稳定后的反馈作为相对控制基准。
        enabled_joint_msg = piper.GetArmJointMsgs()
        enabled_status_msg = piper.GetArmStatus()
        if enabled_joint_msg.Hz <= 0 or enabled_status_msg.Hz <= 0:
            raise RuntimeError("使能后 CAN 反馈频率为零")
        enabled_arm_status = enabled_status_msg.arm_status.arm_status
        if int(enabled_arm_status) != 0:
            raise RuntimeError(f"使能后机械臂状态异常：{enabled_arm_status}")

        target_deg = joint_degrees(enabled_joint_msg)
        control_limits = build_control_limits(target_deg, "使能后位置")
        feedback_publisher.publish(
            float(enabled_joint_msg.time_stamp), target_deg
        )
        print(f"使能后控制基准：{[round(value, 3) for value in target_deg]}")

        terminal_fd = sys.stdin.fileno()
        old_terminal_settings = termios.tcgetattr(terminal_fd)
        tty.setcbreak(terminal_fd)
        print_controls()

        last_feedback_stamp = enabled_joint_msg.time_stamp
        last_feedback_time = time.monotonic()
        last_motion_key_time = 0.0
        motion_pending = False
        active_joint_index = None
        motion_deadline = 0.0
        last_display_time = 0.0

        while True:
            loop_started = time.monotonic()
            joint_msg = piper.GetArmJointMsgs()
            status_msg = piper.GetArmStatus()
            now = time.monotonic()
            feedback_updated = False

            if joint_msg.time_stamp != last_feedback_stamp:
                last_feedback_stamp = joint_msg.time_stamp
                last_feedback_time = now
                feedback_updated = True
            elif now - last_feedback_time > FEEDBACK_TIMEOUT_SECONDS:
                raise RuntimeError("关节反馈中断")

            if joint_msg.Hz <= 0 or status_msg.Hz <= 0:
                raise RuntimeError("CAN 反馈频率为零")

            arm_status = status_msg.arm_status.arm_status
            if int(arm_status) != 0:
                raise RuntimeError(f"机械臂状态异常：{arm_status}")

            actual_deg = joint_degrees(joint_msg)
            if feedback_updated:
                feedback_publisher.publish(
                    float(joint_msg.time_stamp), actual_deg
                )

            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                key = sys.stdin.read(1).lower()

                if key == " ":
                    print("\n收到空格，正在发送软件快速急停...")
                    quick_stop(piper)
                    emergency_stopped = True
                    break

                if key in ("x", "\x1b"):
                    print("\n正在保持当前位置并退出...")
                    hold_current_position(piper, joint_msg)
                    break

                if key in KEY_BINDINGS:
                    last_motion_key_time = now
                    if not motion_pending:
                        joint_index, direction = KEY_BINDINGS[key]
                        candidate = target_deg.copy()
                        candidate[joint_index] += direction * STEP_DEG
                        lower, upper = control_limits[joint_index]

                        if lower <= candidate[joint_index] <= upper:
                            target_deg = candidate
                            motion_pending = True
                            active_joint_index = joint_index
                            motion_deadline = now + MOTION_TIMEOUT_SECONDS
                            print(
                                f"\nJ{joint_index + 1} 新目标："
                                f"{target_deg[joint_index]:.3f}°"
                            )
                        else:
                            print(
                                f"\n拒绝：J{joint_index + 1} 目标 "
                                f"{candidate[joint_index]:.3f}° 超出 "
                                f"[{lower:.1f}, {upper:.1f}]°"
                            )

            # A key press may have changed both the target and the active joint.
            # Calculate the error afterwards so the new motion cannot be marked
            # complete using the previous loop's zero/old error.
            active_error_deg = (
                abs(
                    actual_deg[active_joint_index]
                    - target_deg[active_joint_index]
                )
                if active_joint_index is not None
                else 0.0
            )

            if (
                motion_pending
                and active_error_deg <= TARGET_TOLERANCE_DEG
                and now - last_motion_key_time >= KEY_RELEASE_GAP_SECONDS
            ):
                motion_pending = False
                print("\n目标已到达，可以输入下一次关节按键。")
                active_joint_index = None

            if motion_pending and now > motion_deadline:
                raise TimeoutError(
                    f"J{active_joint_index + 1} 在 "
                    f"{MOTION_TIMEOUT_SECONDS:.0f} 秒内没有到达目标"
                )

            piper.MotionCtrl_2(
                0x01,
                0x01,
                SPEED_PERCENT,
                MOVE_J_MIT_MODE,
            )
            piper.JointCtrl(*sdk_target(target_deg))

            if now - last_display_time >= 0.2:
                active_text = (
                    f"J{active_joint_index + 1}误差={active_error_deg:.2f}°"
                    if active_joint_index is not None
                    else "待命"
                )
                print(
                    f"\r当前={['%.1f' % value for value in actual_deg]} "
                    f"目标={['%.1f' % value for value in target_deg]} "
                    f"{active_text}",
                    end="",
                    flush=True,
                )
                last_display_time = now

            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, LOOP_PERIOD_SECONDS - elapsed))

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C。")
        if control_started and not emergency_stopped:
            print("正在发送软件快速急停...")
            quick_stop(piper)
            emergency_stopped = True

    except Exception:
        if control_started and not emergency_stopped:
            print("\n程序异常，正在发送软件快速急停...")
            quick_stop(piper)
        raise

    finally:
        if old_terminal_settings is not None:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            termios.tcsetattr(
                sys.stdin.fileno(), termios.TCSADRAIN, old_terminal_settings
            )
        try:
            piper.DisconnectPort()
        finally:
            feedback_publisher.close()

    if emergency_stopped:
        print("已完成软件急停尝试；必要时请使用实体急停。")
    else:
        print("键盘控制已结束，机械臂保持使能。")


if __name__ == "__main__":
    main()
