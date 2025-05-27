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
    shoulder_yaw = "left_shoulder_yaw"
    shoulder_pitch = "left_shoulder_pitch"
    elbow_pitch = "left_elbow_yaw"  # FIXME: yaw vs pitch


class RightArm(Node):
    shoulder_yaw = "right_shoulder_yaw"
    shoulder_pitch = "right_shoulder_pitch"
    elbow_pitch = "right_elbow_yaw"  # FIXME: yaw vs pitch


class LeftLeg(Node):
    hip_roll = "left_hip_roll"
    hip_yaw = "left_hip_yaw"
    hip_pitch = "left_hip_pitch"
    knee_pitch = "left_knee_pitch"
    ankle_pitch = "left_ankle_pitch"


class RightLeg(Node):
    hip_roll = "right_hip_roll"
    hip_yaw = "right_hip_yaw"
    hip_pitch = "right_hip_pitch"
    knee_pitch = "right_knee_pitch"
    ankle_pitch = "right_ankle_pitch"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    height = 0.21

    # height = 0.205
    rotation = [0, 0, 0.707, 0.707]

    # left_arm = LeftArm()
    # right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # Legs
            cls.legs.left.hip_pitch: 0.52,
            cls.legs.left.knee_pitch: -1,
            cls.legs.left.hip_yaw: 0,
            cls.legs.left.hip_roll: 0,
            cls.legs.left.ankle_pitch: -0.52,
            cls.legs.right.hip_pitch: -0.52,
            cls.legs.right.knee_pitch: 1,
            cls.legs.right.ankle_pitch: 0.52,
            cls.legs.right.hip_yaw: 0,
            cls.legs.right.hip_roll: 0,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # # left arm
            # Robot.left_arm.shoulder_pitch: {
            #     "lower": -1.7453293,
            #     "upper": 1.7453293,
            # },
            # Robot.left_arm.shoulder_yaw: {
            #     "lower": -0.43633231,
            #     "upper": 1.5707963,
            # },
            # Robot.left_arm.elbow_pitch: {
            #     "lower": -1.5707963,
            #     "upper": 1.5707963,
            # },
            # # Right Arm
            # Robot.right_arm.shoulder_pitch: {
            #     "lower": -1.7453293,
            #     "upper": 1.7453293,
            # },
            # Robot.right_arm.shoulder_yaw: {
            #     "lower": -1.134464,
            #     "upper": 0.87266463,
            # },
            # Robot.right_arm.elbow_pitch: {
            #     "lower": -1.5707963,
            #     "upper": 1.5707963,
            # },
            # Left Leg
            Robot.legs.left.hip_pitch: {
                "lower": -1.4,
                "upper": 1.3,
            },
            Robot.legs.left.hip_yaw: {
                "lower": -1.4,
                "upper": 0.26,
            },
            Robot.legs.left.hip_roll: {
                "lower": -0.70,
                "upper": 0.70,
            },
            Robot.legs.left.knee_pitch: {
                "lower": -1.4,
                "upper": 0,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": -1.5,
                "upper": 1.5,
            },
            # Right Leg
            Robot.legs.right.hip_pitch: {
                "lower": -1.3,
                "upper": 1.4,
            },
            Robot.legs.right.hip_yaw: {
                "lower": -0.26,
                "upper": 1.4,
            },
            Robot.legs.right.hip_roll: {
                "lower": -0.7,
                "upper": 0.7,
            },
            Robot.legs.right.knee_pitch: {
                "lower": 0,
                "upper": 1.4,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -1.5,
                "upper": 1.5,
            },
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 50,
            "hip_yaw": 50,
            "hip_roll": 50,
            "knee_pitch": 50,
            "ankle_pitch": 50,
            # "shoulder_pitch": 5,
            # "shoulder_yaw": 5,
            # "shoulder_roll": 5,
            # "elbow_pitch": 5,
            # "elbow_yaw": 5,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 10,
            "hip_yaw": 10,
            "hip_roll": 10,
            "knee_pitch": 10,
            "ankle_pitch": 10,
            # "shoulder_pitch": 0.3,
            # "shoulder_yaw": 0.3,
            # "shoulder_roll": 0.3,
            # "elbow_pitch": 0.3,
            # "elbow_yaw": 0.3,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 4.9,
            "hip_yaw": 4.9,
            "hip_roll": 4.9,
            "knee_pitch": 4.9,
            "ankle_pitch": 4.9,
            # "shoulder_pitch": 1,
            # "shoulder_yaw": 1,
            # "shoulder_roll": 1,
            # "elbow_pitch": 1,
            # "elbow_yaw": 1,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 7.8,
            "hip_yaw": 7.8,
            "hip_roll": 7.8,
            "knee_pitch": 7.8,
            "ankle_pitch": 7.8,
            # "shoulder_pitch": 10,
            # "shoulder_yaw": 10,
            # "shoulder_roll": 10,
            # "elbow_pitch": 10,
            # "elbow_yaw": 10,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            # pfb30 todo
            "hip_pitch": 0.05,
            "hip_yaw": 0.05,
            "hip_roll": 0.05,
            "knee_pitch": 0.05,
            "ankle_pitch": 0.05,
            # "ankle_pitch": 0.05,
            # "elbow_yaw": 0.05,
            # "elbow_pitch": 0.05,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
