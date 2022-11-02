import numpy as np
import networkx as nx
import torch

def generate_leaves(gym, asset):
    G = nx.DiGraph()
    # Setup up tree handles
    tree_rigid_body_names = gym.get_asset_rigid_body_names(asset)
    for tree_rigid_body_name in tree_rigid_body_names:   
        if any(name in tree_rigid_body_name for name in ['link_']):
            edge = [int(s) for s in tree_rigid_body_name.split('_') if s.isdigit()]
            G.add_edge(edge[0], edge[1])
    print(nx.adjacency_matrix(G).todense())
    print(nx.to_dict_of_dicts(G))

def apply_batch_vector_transform(vec1, vec2):
    """
    Apply transformation on batch of [x,y,z,qx,qy,qz,qw] transformations. 

    Args:
        vec1: x, y, z, qx, qy, qz, qw,
            as tensor of shape (..., 7).
        vec2: x, y, z, qx, qy, qz, qw,
            as tensor of shape (..., 7).

    Returns:
        Vector transformation as tensor of shape (..., 7).
    """
    tf1 = _vector_to_tf(vec1)
    tf2 = _vector_to_tf(vec2)
    tf = torch.einsum('bij,bjk->bik', tf1, tf2)
    vec = _tf_to_vector(tf)
    return vec

def _quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _vector_to_tf(vector):
    if vector.dim()<2:
        vector = vector.unsqueeze(0)
    position = vector[:, :3]
    quaternion = vector[:, 3:]
    matrix = _quaternion_to_matrix(quaternion)
    tf = torch.eye(4, device=matrix.device).reshape((1, 4, 4)).repeat(matrix.size(dim=0), 1, 1)
    tf[:, :3, :3] = matrix
    tf[:, :3, 3] = position
    return tf

def _tf_to_vector(tf):
    position = tf[:, :3, 3]
    quaternion = _matrix_to_quaternion(tf[:, :3, :3])
    return torch.cat((position, quaternion), dim=-1)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    final_quat = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return torch.index_select(final_quat, 1, torch.tensor([1,2,3,0], device=final_quat.device))

