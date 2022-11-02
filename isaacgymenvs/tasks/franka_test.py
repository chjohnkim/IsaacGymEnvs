# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils import tree_utils

import sys
import matplotlib.pyplot as plt

def printt(text):
    print(text)
    sys.exit()

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaTest(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.use_camera = False
        self.display_camera = False
        if self.use_camera and self.display_camera:
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            ax1.title.set_text('Third-Person View')
            ax2.title.set_text('Egocentric View')
            self.im1 = ax1.imshow(np.zeros((128,128,3)))
            self.im2 = ax2.imshow(np.zeros((128,128,3)))
            plt.ion()

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        #self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 26
        # obs include: eef_pose (7) + q_gripper (2) + target (3) + tree (93)
        self.cfg["env"]["numObservations"] = 105 if self.control_type == "osc" else 26
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        
        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 70000000.0
                franka_dof_props['damping'][i] = 5000000.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        franka_start_pose.r = gymapi.Quat(0.0, -0.7071, 0.0, 0.7071)

        # load tree asset
        tree_asset_file = "urdf/trees/tree_0.urdf"
        tree_asset = self.gym.load_asset(self.sim, asset_root, tree_asset_file, asset_options)
        self.num_tree_bodies = self.gym.get_asset_rigid_body_count(tree_asset)
        self.num_tree_dofs = self.gym.get_asset_dof_count(tree_asset)
        tree_dof_props = self.gym.get_asset_dof_properties(tree_asset)
        tree_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) # DOF_MODE_EFFORT (effort) vs DOF_MODE_POS (PD: stiffness, damping)
        for i in range(self.num_tree_dofs):
            tree_dof_props['stiffness'][i] = 10000.0
            tree_dof_props['damping'][i] = 10.0
            tree_dof_props['effort'][i] = 1000.0
        # Define start pose for tree
        tree_start_pose = gymapi.Transform()
        tree_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        tree_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        #tree_utils.generate_leaves(self.gym, tree_asset)

        # Camera sensor properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = 128
        camera_props.height = 128
        # Create camera asset
        camera_size = 0.1
        camera_opts = gymapi.AssetOptions()
        camera_opts.fix_base_link = True
        camera_asset = self.gym.create_box(self.sim, *[camera_size, camera_size, camera_size], camera_opts)
        camera_color = gymapi.Vec3(0.0, 0.0, 1.0)
        # Define start pose for third-person view camera
        tp_camera_pose = gymapi.Transform()
        tp_camera_pose.p = gymapi.Vec3(*[0.0,0.0,2.0])
        tp_camera_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        # Define start pose for egocentric view camera
        self.ego_camera_pose = gymapi.Transform()
        self.ego_camera_pose.p = gymapi.Vec3(*[0.1,0.0,0.0])
        self.ego_camera_pose.r = gymapi.Quat(0.0, -0.7071, 0.0, 0.7071)
        self.ego_camera_pose_vec = torch.Tensor([self.ego_camera_pose.p.x, self.ego_camera_pose.p.y, self.ego_camera_pose.p.z, 
                                                 self.ego_camera_pose.r.x, self.ego_camera_pose.r.y, self.ego_camera_pose.r.z, self.ego_camera_pose.r.w]).to(self.device)

        # Create red cube target asset
        target_pos = [-1.3, 0.0, 2.0]
        target_size = 0.1
        target_opts = gymapi.AssetOptions()
        target_opts.fix_base_link = True
        target_asset = self.gym.create_box(self.sim, *[target_size, target_size, target_size], target_opts)
        target_color = gymapi.Vec3(1.0, 0.0, 0.0)
        # Define start pose for target
        target_start_pose = gymapi.Transform()
        target_start_pose.p = gymapi.Vec3(*target_pos)
        target_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_tree_bodies = self.gym.get_asset_rigid_body_count(tree_asset)
        num_tree_shapes = self.gym.get_asset_rigid_shape_count(tree_asset)
        max_agg_bodies = num_franka_bodies + num_tree_bodies + 3     # 1 for target, tp_camera, ego_camera
        max_agg_shapes = num_franka_shapes + num_tree_shapes + 3     # 1 for target, tp_camera, ego_camera

        self.frankas = []
        self.envs = []
        self.cameras = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1], 1.0)

            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create target and camera actors
            target_actor = self.gym.create_actor(env_ptr, target_asset, target_start_pose, "target", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, target_actor, 0, gymapi.MESH_VISUAL, target_color)
            tp_camera_actor = self.gym.create_actor(env_ptr, camera_asset, tp_camera_pose, "tp_camera", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, tp_camera_actor, 0, gymapi.MESH_VISUAL, camera_color)
            self.ego_camera_actor = self.gym.create_actor(env_ptr, camera_asset, tp_camera_pose, "ego_camera", i+self.num_envs, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, self.ego_camera_actor, 0, gymapi.MESH_VISUAL, camera_color)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create trees
            tree_actor = self.gym.create_actor(env_ptr, tree_asset, tree_start_pose, "tree", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, tree_actor, tree_dof_props)
            
            # Create camera sensors 
            tp_camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_transform(tp_camera_handle, env_ptr, tp_camera_pose)
            ego_camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            panda_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
            self.gym.attach_camera_to_body(ego_camera_handle, env_ptr, panda_handle, self.ego_camera_pose, gymapi.FOLLOW_TRANSFORM)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.cameras.append({'third_person_view': tp_camera_handle, 'egocentric_view': ego_camera_handle})

        # Setup data
        self.init_data()

    def init_data(self):

        # Setup franka sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            
        }
        
        # Setup up tree handles
        tree_handle = 4
        tree_rigid_body_names = self.gym.get_actor_rigid_body_names(env_ptr, tree_handle)
        self.tree_node_names = []
        for tree_rigid_body_name in tree_rigid_body_names:   
            if any(name in tree_rigid_body_name for name in ['_base', '_tip']):
                self.tree_node_names.append(tree_rigid_body_name)
        self.tree_node_handles = {}
        for node_name in self.tree_node_names:
            self.tree_node_handles.update({node_name: self.gym.find_actor_rigid_body_handle(env_ptr, tree_handle, node_name)})

        # Setup target handles
        target_handle = 1
        self.target_handle = self.gym.find_actor_rigid_body_handle(env_ptr, target_handle, "box")

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._hand_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        self.tree_node_states = {}
        for node_name in self.tree_node_names:
            self.tree_node_states.update({node_name: self._rigid_body_state[:, self.tree_node_handles[node_name], :]})
        self.target_state = self._rigid_body_state[:, self.target_handle, :]
        
        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1) # 6 is the number of objects in sim

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :7],
            "q_gripper": self._q[:, 7:9],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # Target
            "target_pos": self.target_state[:, :3],
        })
        for tree_node_name in self.tree_node_names:
            self.states.update({tree_node_name: self.tree_node_states[tree_node_name][:, :3]})

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        if self.use_camera:
            self.gym.render_all_camera_sensors(self.sim)
        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        if not self.use_camera:
            self.num_target_pixels = torch.zeros(self.num_envs)
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.num_target_pixels, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        obs = ["eef_pos", "eef_quat", "target_pos"] + self.tree_node_names
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        #self.obs_buf = self.states
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
        
        if self.use_camera:
            self.gym.start_access_image_tensors(self.sim)
            self.num_target_pixels = torch.zeros(self.num_envs)
            for i in range(self.num_envs):
                tp_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.cameras[i]['third_person_view'], gymapi.IMAGE_COLOR)
                ego_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.cameras[i]['egocentric_view'], gymapi.IMAGE_COLOR)

                torch_tp_camera_tensor = gymtorch.wrap_tensor(tp_camera_tensor)
                torch_ego_camera_tensor = gymtorch.wrap_tensor(ego_camera_tensor)

                r_img = torch_tp_camera_tensor[:,:,0]>100
                sum_img = torch.sum(torch_tp_camera_tensor[:,:,1:3], 2)<50
                num_target_pixels = torch.sum(torch.logical_and(r_img, sum_img)).float()
                self.num_target_pixels[i] = num_target_pixels
            self.gym.end_access_image_tensors(self.sim)
        
            if self.display_camera:
                self.im1.set_data(torch_tp_camera_tensor[:,:,:3].cpu())
                self.im2.set_data(torch_ego_camera_tensor[:,:,:3].cpu())
                plt.pause(1e-6)
        return self.obs_buf

    def reset_idx(self, env_ids):

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        
        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]
        
        # Reset the internal obs accordingly
        self._q[env_ids, :] = torch.zeros_like(self._q[env_ids])
        self._q[env_ids, :9] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = torch.zeros_like(self._pos_control[env_ids])
        self._pos_control[env_ids, :9] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(self._pos_control[env_ids])

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids][:,[0,4]].flatten() # 0 for robot, 4 for tree
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))


        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

        # Update cube states
        #self._root_state[:,3,:7] = self._hand_state[:, :7]
        self._root_state[:,3,:7] = tree_utils.apply_batch_vector_transform(self._hand_state[:, :7], self.ego_camera_pose_vec)
        multi_env_ids_cubes_int32 = self._global_indices[:, 3].flatten().contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        #self.gym.set_actor_rigid_body_states()

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, num_target_pixels, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Dict[str, float], float) -> Tuple[Tensor, Tensor]
    rewards = -(states["eef_lf_pos"][...,0]+states["eef_lf_pos"][...,0])/2
    #rewards = num_target_pixels
    # Compute resets
    #reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

