"""Tests for manifold.py — parallel-flow header pipe + tee branches."""

from __future__ import annotations

import math

import pytest

from cad.cad_config import StackCADConfig


class TestBuildHeader:
    def test_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.manifold import build_header

        result = build_header(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_length_matches_stack(self, default_config: StackCADConfig) -> None:
        """Header pipe length should match stack length."""
        from cad.components.manifold import build_header

        result = build_header(default_config)
        bb = result.val().BoundingBox()
        stack_len_mm = default_config.stack_length * 1000
        assert bb.zlen == pytest.approx(stack_len_mm, rel=0.05)

    def test_is_hollow(self, default_config: StackCADConfig) -> None:
        from cad.components.manifold import build_header

        result = build_header(default_config)
        spec = default_config.manifold
        od = spec.header_od * 1000
        stack_len = default_config.stack_length * 1000
        solid_vol = math.pi * (od / 2) ** 2 * stack_len
        assert result.val().Volume() < solid_vol


class TestBuildTee:
    def test_returns_solid(self, default_config: StackCADConfig) -> None:
        from cad.components.manifold import build_tee

        result = build_tee(default_config)
        assert result is not None
        assert len(result.solids().vals()) >= 1

    def test_has_branch(self, default_config: StackCADConfig) -> None:
        """Tee body should be wider than a simple cylinder."""
        from cad.components.manifold import build_tee

        result = build_tee(default_config)
        bb = result.val().BoundingBox()
        spec = default_config.manifold
        # The tee extends in at least 2 dimensions
        header_od = spec.header_od * 1000
        # Y or X extent should exceed header OD due to branch
        assert max(bb.xlen, bb.ylen) > header_od
