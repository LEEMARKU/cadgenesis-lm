"""cadgenesis.cad.mechanisms
===========================
Kinematic mechanisms: joints and mobility, gears and gear trains, cam
profiles, four-bar linkages, and standard machine parts (bearings, shafts).
"""

from cadgenesis.cad.mechanisms.cams import MOTION_LAWS, CamProfile, CamSegment
from cadgenesis.cad.mechanisms.gears import GearPair, GearTrain, SpurGear, gear_ratio
from cadgenesis.cad.mechanisms.joints import JOINT_TYPES, Joint, Mechanism
from cadgenesis.cad.mechanisms.linkages import FourBarLinkage
from cadgenesis.cad.mechanisms.parts import BEARING_TYPES, Bearing, Shaft

__all__ = [
    "BEARING_TYPES",
    "JOINT_TYPES",
    "MOTION_LAWS",
    "Bearing",
    "CamProfile",
    "CamSegment",
    "FourBarLinkage",
    "GearPair",
    "GearTrain",
    "Joint",
    "Mechanism",
    "Shaft",
    "SpurGear",
    "gear_ratio",
]
