"""Tests for cadgenesis.cad.mechanisms (joints, gears, cams, linkages, parts)."""

from __future__ import annotations

import math

import pytest

from cadgenesis.cad.mechanisms.cams import CamProfile, CamSegment
from cadgenesis.cad.mechanisms.gears import GearPair, GearTrain, SpurGear, gear_ratio
from cadgenesis.cad.mechanisms.joints import Joint, Mechanism
from cadgenesis.cad.mechanisms.linkages import FourBarLinkage
from cadgenesis.cad.mechanisms.parts import Bearing, Shaft


class TestJoints:
    def test_joint_dof(self) -> None:
        revolute = Joint("r", "REVOLUTE", "a", "b")
        spherical = Joint("s", "SPHERICAL", "a", "b")
        assert revolute.dof == 1
        assert spherical.dof == 3

    def test_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            Joint("x", "NOT_A_JOINT", "a", "b")

    def test_mechanism_mobility(self) -> None:
        mechanism = Mechanism("fourbar")
        for link in ("ground", "crank", "coupler", "rocker"):
            mechanism.add_link(link)
        mechanism.add_joint(Joint("j1", "REVOLUTE", "ground", "crank"))
        mechanism.add_joint(Joint("j2", "REVOLUTE", "crank", "coupler"))
        mechanism.add_joint(Joint("j3", "REVOLUTE", "coupler", "rocker"))
        mechanism.add_joint(Joint("j4", "REVOLUTE", "rocker", "ground"))
        assert mechanism.mobility_planar() == 1


class TestGears:
    def test_spur_geometry(self) -> None:
        gear = SpurGear("gear", module=2.0, teeth=20)
        assert gear.pitch_diameter == pytest.approx(40.0)
        assert gear.outer_diameter == pytest.approx(44.0)
        assert gear.circular_pitch == pytest.approx(2 * math.pi)

    def test_gear_pair_ratio(self) -> None:
        driver = SpurGear("d", module=1.0, teeth=20)
        driven = SpurGear("e", module=1.0, teeth=40)
        pair = GearPair(driver, driven)
        assert pair.ratio == 2.0
        assert pair.centre_distance == pytest.approx(30.0)

    def test_invalid_teeth(self) -> None:
        with pytest.raises(ValueError):
            SpurGear("g", module=1.0, teeth=2)

    def test_involute_points(self) -> None:
        gear = SpurGear("g", module=2.0, teeth=20)
        points = gear.involute_points(samples=8)
        assert len(points) == 9
        assert all(math.isfinite(x) and math.isfinite(y) for x, y in points)

    def test_gear_train(self) -> None:
        train = GearTrain()
        train.add_stage(SpurGear("d1", 1.0, 20), SpurGear("e1", 1.0, 40))
        train.add_stage(SpurGear("d2", 1.0, 10), SpurGear("e2", 1.0, 30))
        # output speed relative to input = (20/40) * (10/30)
        assert train.total_ratio() == pytest.approx((20 / 40) * (10 / 30))

    def test_gear_ratio_helper(self) -> None:
        assert gear_ratio(20, 60) == 3.0


class TestCams:
    def test_rise_dwell_fall(self) -> None:
        cam = CamProfile(base_radius=20)
        cam.add_rise_dwell_fall(rise=10, rise_span=120, dwell_span=60, fall_span=180)
        assert cam.max_rise() == 10
        assert cam.pitch_radius_at(60) >= 20
        assert cam.profile_points(samples=36)

    def test_harmonic_displacement(self) -> None:
        segment = CamSegment(0, 120, rise=10, law="harmonic")
        assert segment.displacement(60) == pytest.approx(5.0, abs=1e-6)

    def test_invalid_law(self) -> None:
        with pytest.raises(ValueError):
            CamSegment(0, 10, rise=5, law="bogus")

    def test_overlapping_segments(self) -> None:
        cam = CamProfile(10)
        cam.add_segment(CamSegment(0, 90, 5))
        with pytest.raises(ValueError):
            cam.add_segment(CamSegment(45, 135, 3))


class TestLinkages:
    def test_grashof(self) -> None:
        linkage = FourBarLinkage(ground=60, crank=20, coupler=70, rocker=40)
        assert linkage.is_grashof is True
        assert linkage.mechanism_type == "crank-rocker"

    def test_rocker_angle(self) -> None:
        linkage = FourBarLinkage(ground=60, crank=20, coupler=70, rocker=40)
        angle = linkage.rocker_angle(45)
        assert angle is not None
        assert 0 <= angle < 360

    def test_non_grashof(self) -> None:
        linkage = FourBarLinkage(ground=100, crank=10, coupler=30, rocker=15)
        assert linkage.is_grashof is False

    def test_sweep_angle(self) -> None:
        linkage = FourBarLinkage(ground=60, crank=20, coupler=70, rocker=40)
        assert linkage.sweep_angle() > 0

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError):
            FourBarLinkage(ground=-1, crank=1, coupler=1, rocker=1)


class TestParts:
    def test_bearing(self) -> None:
        bearing = Bearing("b", "ball_radial", bore_mm=10, outer_diameter_mm=30, width_mm=9)
        assert bearing.outer_diameter_mm == 30

    def test_invalid_bearing_type(self) -> None:
        with pytest.raises(ValueError):
            Bearing("b", "not_a_bearing", 10, 30, 9)

    def test_shaft(self) -> None:
        shaft = Shaft("s", diameter_mm=20, length_mm=100)
        assert shaft.volume_mm3() == pytest.approx(math.pi * 100 * 100)

    def test_shaft_invalid_journal(self) -> None:
        with pytest.raises(ValueError):
            Shaft("s", diameter_mm=20, length_mm=100, journal_positions_mm=[150])
