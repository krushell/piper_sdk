from __future__ import annotations

import torch


@torch.jit.script
def quat_from_euler_xyz(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Convert XYZ Euler angles in radians to quaternions in ``(w, x, y, z)`` order."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack((qw, qx, qy, qz), dim=-1)


def quat_mul(quat_1: torch.Tensor, quat_2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in ``(w, x, y, z)`` order."""
    if quat_1.shape != quat_2.shape or quat_1.shape[-1] != 4:
        raise ValueError("Quaternion inputs must have matching shape (..., 4).")

    w1, x1, y1, z1 = torch.unbind(quat_1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(quat_2, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_unique(quat: torch.Tensor) -> torch.Tensor:
    """Return the equivalent quaternion whose real component is non-negative."""
    return torch.where(quat[..., 0:1] < 0, -quat, quat)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Return rotation matrices for rotations about one Cartesian axis."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        rotation_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        rotation_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        rotation_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("axis must be one of 'X', 'Y', or 'Z'")

    return torch.stack(rotation_flat, dim=-1).reshape(angle.shape + (3, 3))


def matrix_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
    """Convert ``(roll, pitch, yaw)`` angles in radians to rotation matrices."""
    if rpy.dim() == 0 or rpy.shape[-1] != 3:
        raise ValueError("Invalid RPY angles; expected shape (..., 3).")

    roll, pitch, yaw = torch.unbind(rpy, dim=-1)
    rotation_x = _axis_angle_rotation("X", roll)
    rotation_y = _axis_angle_rotation("Y", pitch)
    rotation_z = _axis_angle_rotation("Z", yaw)
    return torch.matmul(torch.matmul(rotation_z, rotation_y), rotation_x)
