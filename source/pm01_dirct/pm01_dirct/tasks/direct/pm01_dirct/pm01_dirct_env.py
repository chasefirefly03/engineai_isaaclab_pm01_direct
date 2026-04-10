# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

from .pm01_dirct_env_cfg import Pm01DirctFaltEnvCfg


def gait_phase(env:DirectRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


class Pm01DirctEnv(DirectRLEnv):
    cfg: Pm01DirctFaltEnvCfg

    def __init__(self, cfg: Pm01DirctFaltEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                # "undesired_contacts",
                "flat_orientation_l2",
                "feet_slide",
                "joint_deviation_hip",
                "joint_deviation_knee",
                "joint_deviation_arms",
                "joint_deviation_torso",
                "stand_still",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("link_base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(["link_ankle_roll_r", "link_ankle_roll_l"])
        self._hip_joints = [".*_hip_yaw_.*", ".*_hip_roll_.*"]
        self._knee_joints = [".*_knee_pitch_.*"]
        self._arm_joints = [".*_shoulder_.*", ".*_elbow_.*"]
        self._torso_joints = ["j12_waist_yaw"]
        # self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos


    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.root_quat_w,
                    self._commands,
                    self._robot.data.projected_gravity_b,
                    gait_phase(self, period=0.6),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot. data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1) * (
            # torch.norm(self._commands[:, :2], dim=1) > 0.1
            torch.norm(self._commands[:, :3], dim=1) > 0.1
        )
        # undesired contacts
        # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # is_contact = (
        #     torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        # )
        # undesired_contacts = torch.sum(is_contact, dim=1)   
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)      
        #feet_slide
        asset: Articulation = self.scene["robot"]
        feet_contacts = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        body_vel = asset.data.body_lin_vel_w[:, self._feet_ids, :2]
        feet_slide = torch.sum(body_vel.norm(dim=-1) * feet_contacts, dim=1)
        #joint_deviation_hip
        hip_asset_cfg = SceneEntityCfg("robot", joint_names=self._hip_joints)
        hip_asset: Articulation = self.scene[hip_asset_cfg.name]
        hip_angle = hip_asset.data.joint_pos[:, hip_asset_cfg.joint_ids] - hip_asset.data.default_joint_pos[:, hip_asset_cfg.joint_ids]
        joint_deviation_hip =  torch.sum(torch.abs(hip_angle), dim=1)
        #joint_deviation_knee
        knee_asset_cfg = SceneEntityCfg("robot", joint_names=self._knee_joints)
        knee_asset: Articulation = self.scene[knee_asset_cfg.name]
        knee_angle = knee_asset.data.joint_pos[:, knee_asset_cfg.joint_ids] - knee_asset.data.default_joint_pos[:, knee_asset_cfg.joint_ids]
        joint_deviation_knee =  torch.sum(torch.abs(knee_angle), dim=1)        
        #joint_deviation_arms
        arm_asset_cfg = SceneEntityCfg("robot", joint_names=self._arm_joints)
        arm_asset: Articulation = self.scene[arm_asset_cfg.name]        
        arm_angle = arm_asset.data.joint_pos[:, arm_asset_cfg.joint_ids] - arm_asset.data.default_joint_pos[:, arm_asset_cfg.joint_ids]
        joint_deviation_arms =  torch.sum(torch.abs(arm_angle), dim=1)        
        #joint_deviation_torso
        torso_asset_cfg = SceneEntityCfg("robot", joint_names=self._torso_joints)
        torso_asset: Articulation = self.scene[torso_asset_cfg.name]          
        torso_angle = torso_asset.data.joint_pos[:, torso_asset_cfg.joint_ids] - torso_asset.data.default_joint_pos[:, torso_asset_cfg.joint_ids]
        joint_deviation_torso =  torch.sum(torch.abs(torso_angle), dim=1)        
        #stand_still
        angle = asset.data.joint_pos - asset.data.default_joint_pos
        joint_deviation_l1 = torch.sum(torch.abs(angle), dim=1)
        # stand_still = joint_deviation_l1 * (torch.norm(self._commands[:, :2], dim=1) < self.cfg.command_threshold)
        stand_still = joint_deviation_l1 * (torch.norm(self._commands[:, :3], dim=1) < self.cfg.command_threshold)
        
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.track_lin_vel_xy_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.track_ang_vel__reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.lin_vel_z_l2_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            # "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "feet_slide": feet_slide * self.cfg.feet_slide_reward_scale * self.step_dt,
            "joint_deviation_hip": joint_deviation_hip * self.cfg.joint_deviation_hip_reward_scale * self.step_dt,
            "joint_deviation_knee": joint_deviation_knee * self.cfg.joint_deviation_knee_reward_scale * self.step_dt,
            "joint_deviation_arms": joint_deviation_arms * self.cfg.joint_deviation_arms_reward_scale * self.step_dt,
            "joint_deviation_torso": joint_deviation_torso * self.cfg.joint_deviation_torso_reward_scale * self.step_dt,
            "stand_still": stand_still * self.cfg.stand_still * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Command
        self._commands[env_ids, 0] = torch.tensor(0.0, device=self.device).uniform_(0.4, 1.0)
        # Set some targets to zero (10% chance)
        zero_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        self._commands[env_ids[zero_mask], 0] = 0.0
        self._commands[env_ids, 2] = torch.tensor(0.0, device=self.device).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

