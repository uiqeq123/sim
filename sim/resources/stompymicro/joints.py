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
    shoulder_pitch = "left shoulder pitch"
    shoulder_yaw = "left shoulder yaw"
    elbow_pitch = "left elbow yaw"


class RightArm(Node):
    shoulder_pitch = "right shoulder pitch"
    shoulder_yaw = "right shoulder yaw"
    elbow_pitch = "right elbow yaw"


class Arms(Node):
    left = LeftArm()
    right = RightArm()


class LeftLeg(Node):
    hip_pitch = "left hip pitch"
    hip_yaw = "left hip yaw"
    hip_roll = "left hip roll"
    knee_pitch = "left knee pitch"
    ankle_pitch = "left ankle pitch"


class RightLeg(Node):
    hip_pitch = "right hip pitch"
    hip_yaw = "right hip yaw"
    hip_roll = "right hip roll"
    knee_pitch = "right knee pitch"
    ankle_pitch = "right ankle pitch"


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
            cls.arms.left.shoulder_pitch: 0.0,
            cls.arms.left.shoulder_yaw: 0.0,
            cls.arms.left.elbow_pitch: 0.0,
            ## Right arm
            cls.arms.right.shoulder_pitch: 0.0,
            cls.arms.right.shoulder_yaw: 0.0,
            cls.arms.right.elbow_pitch: 0.0,
            # Legs
            ## Left leg
            cls.legs.left.hip_pitch: 0.0,
            cls.legs.left.hip_yaw: 0.0,
            cls.legs.left.hip_roll: 0.0,
            cls.legs.left.knee_pitch: 0.0,
            cls.legs.left.ankle_pitch: 0.0,
            ## Right leg
            cls.legs.right.hip_pitch: 0.0,
            cls.legs.right.hip_yaw: 0.0,
            cls.legs.right.hip_roll: 0.0,
            cls.legs.right.knee_pitch: 0.0,
            cls.legs.right.ankle_pitch: 0.0,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
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
            cls.arms.left.shoulder_pitch: 5.0,
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
