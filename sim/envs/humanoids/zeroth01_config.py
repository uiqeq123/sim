"""Defines the environment configuration for the Getting up task"""

import numpy as np

from sim.env import robot_urdf_path
from sim.envs.base.legged_robot_config import (  # type: ignore
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)
from sim.resources.zeroth01.joints import Robot

NUM_JOINTS = len(Robot.all_joints())  # 20


class Zeroth01Cfg(LeggedRobotCfg):
    """Configuration class for the Legs humanoid robot."""

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 11 + NUM_JOINTS * 3
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 25 + NUM_JOINTS * 4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = NUM_JOINTS
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1
        terminate_after_contacts_on = []

    class asset(LeggedRobotCfg.asset):
        name = "zeroth01"
        file = str(robot_urdf_path(name))

        #foot_name = ["foot_left", "foot_right"]
        #knee_name = ["left_knee_pitch_motor", "right_knee_pitch_motor"]
        foot_name = ["left_ankle_pitch", "right_ankle_pitch"]
        knee_name = ["left_knee_pitch", "right_knee_pitch"]


        termination_height = 0.08
        default_feet_height = 0.015

        terminate_after_contacts_on = ["base_link"]

        penalize_contacts_on = ["base_link"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

        # BAM parameters
        # TODO update effort to larger one
        friction = 0.053343597773929877
        armature = 0.008793405204572328

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            lin_vel = 0.05
            ang_vel = 0.1
            quat = 0.03
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, Robot.height]
        # rot = Robot.rotation

        default_joint_angles = {k: 0.0 for k in Robot.all_joints()}

        default_positions = Robot.default_standing()
        for joint in default_positions:
            default_joint_angles[joint] = default_positions[joint]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = Robot.stiffness()
        damping = Robot.damping()
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        start_pos_noise = 0.05
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True  # True
        added_mass_range = [-0.1, 0]
        push_robots = True  # True
        push_interval_s = 4
        max_push_vel_xy = 0.05
        max_push_ang_vel = 0.05
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0, 0.23 ] # min max [m/s]
            lin_vel_y = [0, 0]  # min max [m/s]
            ang_vel_yaw = [0, 0]  # min max [rad/s]
            heading = [0,0]  # min max [rad]

    class rewards:
        base_height_target = Robot.height
        min_dist = 0.08
        max_dist = 0.12

        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.1  # rad
        target_feet_height = 0.015  # m
        cycle_time = 0.64  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5.0
        max_contact_force = 100  # forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1
            feet_contact_number = 1.2
            feet_air_time = 1
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5

            # base pos
            default_joint_pos = 0.5
            orientation = 1
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0

    class viewer:
        ref_env = 0
        pos = [4, -4, 2]
        lookat = [0, -2, 0]


class Zeroth01StandingCfg(Zeroth01Cfg):
    class rewards:
        base_height_target = Robot.height
        # this standing cfg
        min_dist = 0.08
        max_dist = 0.10

        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17*2  # rad
        target_feet_height = 0.03  # m
        cycle_time = 0.4  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5.0
        max_contact_force = 50  # forces above this value are penalized

        class scales:
            # base pos
            default_joint_pos = 1.0
            orientation = 1
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.0


class Zeroth01CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "StompyMicro"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
