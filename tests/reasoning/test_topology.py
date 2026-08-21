"""tests/reasoning/test_topology.py
==================================
Unit tests for cadgenesis.reasoning.topology.
"""

from __future__ import annotations

import pytest

from cadgenesis.reasoning.topology import TopologyAnalyzer, TopologyStats


class TestEuler:
    def test_cube(self):
        # V - E + F = 8 - 12 + 6 = 2
        assert TopologyAnalyzer.euler_characteristic(8, 12, 6) == 2

    def test_genus_zero(self):
        assert TopologyAnalyzer.genus_for_surface(2) == 0

    def test_torus(self):
        # genus 1 => chi = 0
        assert TopologyAnalyzer.genus_for_surface(0) == 1

    def test_genus_invalid(self):
        with pytest.raises(ValueError):
            TopologyAnalyzer.genus_for_surface(3)
        with pytest.raises(ValueError):
            TopologyAnalyzer.genus_for_surface(4)  # chi=4 => genus -1

    def test_negative_counts(self):
        with pytest.raises(ValueError):
            TopologyAnalyzer.euler_characteristic(-1, 0, 0)


class TestComponents:
    def test_two_components(self):
        # vertices 0-3, edges connect {0,1} and {2,3}
        assert TopologyAnalyzer.connected_components(4, [(0, 1), (2, 3)]) == 2

    def test_one_component(self):
        assert TopologyAnalyzer.connected_components(3, [(0, 1), (1, 2)]) == 1

    def test_bad_edge_endpoint(self):
        with pytest.raises(IndexError):
            TopologyAnalyzer.connected_components(3, [(0, 5)])


class TestManifold:
    def test_two_triangles_sharing_edge_is_manifold_open(self):
        faces = [(0, 1, 2), (0, 2, 3)]
        assert TopologyAnalyzer.is_manifold(faces)
        assert not TopologyAnalyzer.is_closed(faces)

    def test_non_manifold_three_faces(self):
        faces = [(0, 1, 2), (0, 2, 3), (0, 2, 4)]
        assert not TopologyAnalyzer.is_manifold(faces)

    def test_tetrahedron_closed(self):
        faces = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
        assert TopologyAnalyzer.is_manifold(faces)
        assert TopologyAnalyzer.is_closed(faces)


class TestAnalyze:
    def test_cube_counts(self):
        stats = TopologyAnalyzer.analyze(vertices=8, edges=12, faces=6, shells=1, solids=1)
        assert isinstance(stats, TopologyStats)
        assert stats.euler_characteristic == 2
        assert stats.genus == 0
        assert stats.is_valid

    def test_inconsistent_gets_note(self):
        stats = TopologyAnalyzer.analyze(vertices=8, edges=10, faces=5, shells=1, solids=1)
        assert not stats.is_valid
        assert any("mismatch" in n for n in stats.notes)

    def test_analyze_mesh_tetrahedron(self):
        faces = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
        stats = TopologyAnalyzer.analyze_mesh(faces)
        assert stats.is_closed
        assert stats.genus == 0
        assert stats.vertices == 4
        assert stats.faces == 4
        assert stats.euler_characteristic == 2

    def test_analyze_mesh_empty(self):
        with pytest.raises(ValueError):
            TopologyAnalyzer.analyze_mesh([])

    def test_summary_dict(self):
        stats = TopologyAnalyzer.analyze(vertices=8, edges=12, faces=6, solids=1)
        summary = stats.summary()
        assert summary["euler_characteristic"] == 2
