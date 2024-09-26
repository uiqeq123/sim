"""Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""

import textwrap
from abc import ABC
from typing import Dict, List, Tuple, Union


class Node(ABC):
    @classmethod
    def children(cls) -> List["Union[Node, str]"]:
        return [
            attr
            for attr in (getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__"))
            if isinstance(attr, (Node, str))
        ]

    @classmethod
    def joints(cls) -> List[str]:
        return [
            attr
            for attr in (getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__"))
            if isinstance(attr, str)
        ]

    @classmethod
    def joints_motors(cls) -> List[Tuple[str, str]]:
        joint_names: List[Tuple[str, str]] = []
        for attr in dir(cls):
            if not attr.startswith("__"):
                attr2 = getattr(cls, attr)
                if isinstance(attr2, str):
                    joint_names.append((attr, attr2))

        return joint_names

    @classmethod
    def all_joints(cls) -> List[str]:
        leaves = cls.joints()
        for child in cls.children():
            if isinstance(child, Node):
                leaves.extend(child.all_joints())
        return leaves

    def __str__(self) -> str:
        parts = [str(child) for child in self.children()]
        parts_str = textwrap.indent("\n" + "\n".join(parts), "  ")
        return f"[{self.__class__.__name__}]{parts_str}"


class LeftArm(Node):
    shoulder_pitch = "left_shoulder_pitch"
    shoulder_yaw = "left_shoulder_yaw"
    elbow_pitch = "left_elbow_yaw"


class RightArm(Node):
    shoulder_pitch = "right_shoulder_pitch"
    shoulder_yaw = "right_shoulder_yaw"
    elbow_pitch = "right_elbow_yaw"


class Arms(Node):
    left = LeftArm()
    right = RightArm()


class LeftLeg(Node):
    hip_pitch = "left_hip_pitch"
    hip_yaw = "left_hip_yaw"
    hip_roll = "left_hip_roll"
    knee_pitch = "left_knee_pitch"
    ankle_pitch = "left_ankle_pitch"


class RightLeg(Node):
    hip_pitch = "right_hip_pitch"
    hip_yaw = "right_hip_yaw"
    hip_roll = "right_hip_roll"
    knee_pitch = "right_knee_pitch"
    ankle_pitch = "right_ankle_pitch"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    height = 0.21
    rotation = [0, 0, 0, 1]

    arms = Arms()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # Arms
            ## Left arm
            cls.arms.left.shoulder_pitch: 2.25,
            cls.arms.left.shoulder_yaw: 1.57,
            cls.arms.left.elbow_pitch: -1.61,
            ## Right arm
            cls.arms.right.shoulder_pitch: 3.45,
            cls.arms.right.shoulder_yaw: -0.1,
            cls.arms.right.elbow_pitch: -1.49,
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: 0.33,
            cls.legs.left.hip_yaw: 4.67,
            cls.legs.left.hip_roll: -1.52,
            cls.legs.left.knee_pitch: -0.61,
            cls.legs.left.ankle_pitch: 1.88,
            ## Right leg
            cls.legs.right.hip_pitch: 2.91,
            cls.legs.right.hip_yaw: 3.22,
            cls.legs.right.hip_roll: 3.24,
            cls.legs.right.knee_pitch: 0.65,
            cls.legs.right.ankle_pitch: -0.54,
        }

    @classmethod
    def default_standing2(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left arm
            Robot.arms.left.shoulder_pitch: {
                "lower": 2.54,
                "upper": 2.56,
            },
            Robot.arms.left.shoulder_yaw: {
                "lower": 1.56,
                "upper": 1.58,
            },
            # Robot.arms.left.shoulder_roll: {
            #     "lower": 3.13,
            #     "upper": 3.14,
            # },
            Robot.arms.left.elbow_pitch: {
                "lower": -1.56,
                "upper": -1.58,
            },
            # Robot.arms.left.hand.wrist_roll: {
            #     "lower": -1.56,
            #     "upper": -1.58,
            # },
            # Robot.arms.left.hand.gripper: {
            #     "lower": 0,
            #     "upper": 1.57,
            # },
            # right arm
            Robot.arms.right.shoulder_pitch: {
                "lower": 3.119,
                "upper": 3.121,
            },
            Robot.arms.right.shoulder_yaw: {
                "lower": 1.981,
                "upper": 1.979,
            },
            # Robot.arms.right.shoulder_roll: {
            #     "lower": -1.381,
            #     "upper": -1.979,
            # },
            Robot.arms.right.elbow_pitch: {
                "lower": -3.319,
                "upper": 3.321,
            },
            # Robot.arms.right.hand.wrist_roll: {
            #     "lower": -0.001,
            #     "upper": 0.001,
            # },
            # Robot.arms.right.hand.gripper: {
            #     "lower": 0,
            #     "upper": 1.57,
            # },
            # left leg
            Robot.legs.left.hip_pitch: {
                "lower": -1.14,
                "upper": 1.14,
            },
            Robot.legs.left.hip_roll: {
                "lower": -3.5,
                "upper": 0.5,
            },
            Robot.legs.left.hip_yaw: {
                "lower": 3.14,
                "upper": 5.14,
            },
            Robot.legs.left.knee_pitch: {
                "lower": -2,
                "upper": 0,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": 1.4,
                "upper": 2.2,
            },
            # right leg
            Robot.legs.right.hip_pitch: {
                "lower": 0.55,
                "upper": 3.55,
            },
            Robot.legs.right.hip_roll: {
                "lower": 2.75,
                "upper": 3.99,
            },
            Robot.legs.right.hip_yaw: {
                "lower": 2.24,
                "upper": 4.24,
            },
            Robot.legs.right.knee_pitch: {
                "lower": 0,
                "upper": 2,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -1.0,
                "upper": 0.2,
            },
        }

    @classmethod
    def default_limits2(cls) -> Dict[str, Dict[str, float]]:
        return {
            # Arms
            ## Left arm
            cls.arms.left.shoulder_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.arms.left.shoulder_yaw: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.arms.left.elbow_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            ## Right arm
            cls.arms.right.shoulder_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.arms.right.shoulder_yaw: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.arms.right.elbow_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.left.hip_yaw: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.left.hip_roll: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.left.knee_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.left.ankle_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            ## Right leg
            cls.legs.right.hip_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.right.hip_yaw: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.right.hip_roll: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.right.knee_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
            cls.legs.right.ankle_pitch: {
                "lower": -1.57,
                "upper": 1.57,
            },
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left arm
            Robot.arms.left.shoulder_pitch: {
                "lower": 2.04,
                "upper": 3.06,
            },
            Robot.arms.left.shoulder_yaw: {
                "lower": -1,
                "upper": 2,
            },
            # Robot.arms.left.shoulder_roll: {
            #     "lower": 2.63,
            #     "upper": 3.64,
            # },
            Robot.arms.left.elbow_pitch: {
                "lower": -2.06,
                "upper": -1.08,
            },
            # Robot.arms.left.hand.wrist_roll: {
            #     "lower": -2.06,
            #     "upper": -1.08,
            # },
            # Robot.arms.left.hand.gripper: {
            #     "lower": -0.5,
            #     "upper": 2.07,
            # },
            # right arm
            Robot.arms.right.shoulder_pitch: {
                "lower": 2.619,
                "upper": 3.621,
            },
            Robot.arms.right.shoulder_yaw: {
                "lower": -1.481,
                "upper": 1,
            },
            # Robot.arms.right.shoulder_roll: {
            #     "lower": -1.881,
            #     "upper": -1.479,
            # },
            Robot.arms.right.elbow_pitch: {
                "lower": -3.819,
                "upper": 3.821,
            },
            # Robot.arms.right.hand.wrist_roll: {
            #     "lower": -0.501,
            #     "upper": 0.501,
            # },
            # Robot.arms.right.hand.gripper: {
            #     "lower": -0.5,
            #     "upper": 2.07,
            # },
            # left leg
            Robot.legs.left.hip_pitch: {
                "lower": -1.64,
                "upper": 1.64,
            },
            Robot.legs.left.hip_roll: {
                "lower": -4.0,
                "upper": 1.0,
            },
            Robot.legs.left.hip_yaw: {
                "lower": 2.64,
                "upper": 5.64,
            },
            Robot.legs.left.knee_pitch: {
                "lower": -2.5,
                "upper": 0.5,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": 0.9,
                "upper": 2.7,
            },
            # right leg
            Robot.legs.right.hip_pitch: {
                "lower": 0.05,
                "upper": 4.05,
            },
            Robot.legs.right.hip_roll: {
                "lower": 2.25,
                "upper": 4.49,
            },
            Robot.legs.right.hip_yaw: {
                "lower": 1.74,
                "upper": 4.74,
            },
            Robot.legs.right.knee_pitch: {
                "lower": -0.5,
                "upper": 2.5,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -1.5,
                "upper": 0.7,
            },
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            # Arms
            ## Left arm
            cls.arms.left.shoulder_pitch: 150,
            cls.arms.left.shoulder_yaw: 45,
            cls.arms.left.elbow_pitch: 45,
            ## Right arm
            cls.arms.right.shoulder_pitch: 150,
            cls.arms.right.shoulder_yaw: 45,
            cls.arms.right.elbow_pitch: 45,
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: 250,
            cls.legs.left.hip_yaw: 250,
            cls.legs.left.hip_roll: 150,
            cls.legs.left.knee_pitch: 250,
            cls.legs.left.ankle_pitch: 150,
            ## Right leg
            cls.legs.right.hip_pitch: 250,
            cls.legs.right.hip_yaw: 250,
            cls.legs.right.hip_roll: 150,
            cls.legs.right.knee_pitch: 250,
            cls.legs.right.ankle_pitch: 150,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            # Arms
            ## Left arm
            cls.arms.left.shoulder_pitch: 10.0,
            cls.arms.left.shoulder_yaw: 10.0,
            cls.arms.left.elbow_pitch: 5.0,
            ## Right arm
            cls.arms.right.shoulder_pitch: 10.0,
            cls.arms.right.shoulder_yaw: 10.0,
            cls.arms.right.elbow_pitch: 5.0,
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: 10.0,
            cls.legs.left.hip_yaw: 10.0,
            cls.legs.left.hip_roll: 10.0,
            cls.legs.left.knee_pitch: 10.0,
            cls.legs.left.ankle_pitch: 10.0,
            ## Right leg
            cls.legs.right.hip_pitch: 10.0,
            cls.legs.right.hip_yaw: 10.0,
            cls.legs.right.hip_roll: 10.0,
            cls.legs.right.knee_pitch: 10.0,
            cls.legs.right.ankle_pitch: 10.0,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            # Arms
            ## Left arm
            cls.arms.left.shoulder_pitch: 80.0,
            cls.arms.left.shoulder_yaw: 80.0,
            cls.arms.left.elbow_pitch: 80.0,
            ## Right arm
            cls.arms.right.shoulder_pitch: 80.0,
            cls.arms.right.shoulder_yaw: 80.0,
            cls.arms.right.elbow_pitch: 80.0,
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: 80.0,
            cls.legs.left.hip_yaw: 80.0,
            cls.legs.left.hip_roll: 80.0,
            cls.legs.left.knee_pitch: 80.0,
            cls.legs.left.ankle_pitch: 80.0,
            ## Right leg
            cls.legs.right.hip_pitch: 80.0,
            cls.legs.right.hip_yaw: 80.0,
            cls.legs.right.hip_roll: 80.0,
            cls.legs.right.knee_pitch: 80.0,
            cls.legs.right.ankle_pitch: 80.0,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            # Arms
            ## Left arm
            cls.arms.left.shoulder_pitch: 5.0,  # TODO: Possibly all 40?
            cls.arms.left.shoulder_yaw: 5.0,
            cls.arms.left.elbow_pitch: 5.0,
            ## Right arm
            cls.arms.right.shoulder_pitch: 5.0,
            cls.arms.right.shoulder_yaw: 5.0,
            cls.arms.right.elbow_pitch: 5.0,
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: 5.0,
            cls.legs.left.hip_yaw: 5.0,
            cls.legs.left.hip_roll: 5.0,
            cls.legs.left.knee_pitch: 5.0,
            cls.legs.left.ankle_pitch: 5.0,
            ## Right leg
            cls.legs.right.hip_pitch: 5.0,
            cls.legs.right.hip_yaw: 5.0,
            cls.legs.right.hip_roll: 5.0,
            cls.legs.right.knee_pitch: 5.0,
            cls.legs.right.ankle_pitch: 5.0,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            # Arms
            ## Left arm
            cls.arms.left.shoulder_pitch: 0.1,
            cls.arms.left.shoulder_yaw: 0.1,
            cls.arms.left.elbow_pitch: 0.08,
            ## Right arm
            cls.arms.right.shoulder_pitch: 0.1,
            cls.arms.right.shoulder_yaw: 0.1,
            cls.arms.right.elbow_pitch: 0.08,
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: 0.15,
            cls.legs.left.hip_yaw: 0.15,
            cls.legs.left.hip_roll: 0.15,
            cls.legs.left.knee_pitch: 0.12,
            cls.legs.left.ankle_pitch: 0.1,
            ## Right leg
            cls.legs.right.hip_pitch: 0.15,
            cls.legs.right.hip_yaw: 0.15,
            cls.legs.right.hip_roll: 0.15,
            cls.legs.right.knee_pitch: 0.12,
            cls.legs.right.ankle_pitch: 0.1,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.resources.stompymicro.joints
    print_joints()
