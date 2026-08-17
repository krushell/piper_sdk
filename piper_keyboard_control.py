#!/usr/bin/env python3
import select
import sys
import termios
import time
import tty

from piper_sdk import C_PiperInterface_V2


CAN_NAME = "can0"
STEP_DEG = 10.0
SPEED_PERCENT = 5
TARGET_TOLERANCE_DEG = 0.5
FEEDBACK_TIMEOUT_SECONDS = 0.5
MOTION_TIMEOUT_SECONDS = 20.0
KEY_RELEASE_GAP_SECONDS = 0.15
LOOP_PERIOD_SECONDS = 0.02
FEEDBACK_LIMIT_MARGIN_DEG = 5.0

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


def hold_current_position(piper, joint_msg):
    target = sdk_target(joint_degrees(joint_msg))
    for _ in range(10):
        piper.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
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


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("该脚本必须在交互式终端中运行")

    piper = C_PiperInterface_V2(CAN_NAME)
    control_started = False
    emergency_stopped = False
    old_terminal_settings = None

    try:
        piper.ConnectPort()

        feedback_deadline = time.monotonic() + 3.0
        while True:
            joint_msg = piper.GetArmJointMsgs()
            status_msg = piper.GetArmStatus()
            if joint_msg.Hz > 0 and status_msg.Hz > 0:
                break
            if time.monotonic() > feedback_deadline:
                raise RuntimeError(
                    "没有收到完整反馈，请检查 CAN、机械臂电源和从臂模式"
                )
            time.sleep(LOOP_PERIOD_SECONDS)

        initial_deg = joint_degrees(joint_msg)
        build_control_limits(initial_deg, "当前位置")

        arm_status = status_msg.arm_status.arm_status
        if int(arm_status) != 0:
            raise RuntimeError(f"机械臂启动前状态异常：{arm_status}")

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
        confirmed_arm_status = confirmed_status_msg.arm_status.arm_status
        if int(confirmed_arm_status) != 0:
            raise RuntimeError(f"机械臂启动前状态异常：{confirmed_arm_status}")

        build_control_limits(confirmed_deg, "确认位置", show_warnings=False)
        control_started = True
        enable_deadline = time.monotonic() + 5.0
        while not piper.EnablePiper():
            if time.monotonic() > enable_deadline:
                raise RuntimeError("机械臂使能失败")
            time.sleep(0.01)

        # 失能状态与使能状态的反馈零位可能不同，以使能后的反馈作为相对控制基准。
        time.sleep(0.1)
        enabled_joint_msg = piper.GetArmJointMsgs()
        enabled_status_msg = piper.GetArmStatus()
        if enabled_joint_msg.Hz <= 0 or enabled_status_msg.Hz <= 0:
            raise RuntimeError("使能后 CAN 反馈频率为零")
        enabled_arm_status = enabled_status_msg.arm_status.arm_status
        if int(enabled_arm_status) != 0:
            raise RuntimeError(f"使能后机械臂状态异常：{enabled_arm_status}")

        target_deg = joint_degrees(enabled_joint_msg)
        control_limits = build_control_limits(target_deg, "使能后位置")
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

            actual_deg = joint_degrees(joint_msg)

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

            piper.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
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
        piper.DisconnectPort()

    if emergency_stopped:
        print("已完成软件急停尝试；必要时请使用实体急停。")
    else:
        print("键盘控制已结束，机械臂保持使能。")


if __name__ == "__main__":
    main()
