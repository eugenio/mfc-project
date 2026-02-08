"""Tests for utils/plotting.py - targeting 98%+ coverage."""
import os
import sys
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from utils.plotting import (
    SubplotLabeler,
    add_horizontal_line,
    add_subplot_label,
    add_text_annotation,
    create_labeled_subplots,
    create_standard_mfc_plots,
    plot_time_series,
    save_figure,
    setup_axis,
)


class TestSubplotLabeler:
    """Tests for SubplotLabeler class."""

    def test_initial_label(self):
        labeler = SubplotLabeler()
        assert labeler.next_label() == "a"

    def test_sequential_labels(self):
        labeler = SubplotLabeler()
        labels = [labeler.next_label() for _ in range(5)]
        assert labels == ["a", "b", "c", "d", "e"]

    def test_all_26_letters(self):
        labeler = SubplotLabeler()
        labels = [labeler.next_label() for _ in range(26)]
        assert labels[0] == "a"
        assert labels[25] == "z"

    def test_double_letter_labels(self):
        labeler = SubplotLabeler()
        # Skip first 26 labels
        for _ in range(26):
            labeler.next_label()
        # Next should be "aa"
        assert labeler.next_label() == "aa"
        assert labeler.next_label() == "ab"

    def test_beyond_zz_labels(self):
        labeler = SubplotLabeler()
        # Skip to beyond aa-zz range (26 + 26*26 = 702)
        for _ in range(702):
            labeler.next_label()
        label = labeler.next_label()
        assert label.startswith("subplot_")

    def test_reset(self):
        labeler = SubplotLabeler()
        labeler.next_label()
        labeler.next_label()
        labeler.reset()
        assert labeler.next_label() == "a"


class TestCreateLabeledSubplots:
    """Tests for create_labeled_subplots."""

    def test_single_subplot(self):
        fig, axes, labeler = create_labeled_subplots(1, 1)
        assert len(axes) == 1
        plt.close(fig)

    def test_2x2_subplots(self):
        fig, axes, labeler = create_labeled_subplots(2, 2)
        assert len(axes) == 4
        plt.close(fig)

    def test_with_title(self):
        fig, axes, labeler = create_labeled_subplots(1, 2, title="Test Title")
        assert len(axes) == 2
        plt.close(fig)

    def test_custom_figsize(self):
        fig, axes, labeler = create_labeled_subplots(1, 1, figsize=(6, 4))
        plt.close(fig)

    def test_3x1_subplots(self):
        fig, axes, labeler = create_labeled_subplots(3, 1)
        assert len(axes) == 3
        plt.close(fig)


class TestAddSubplotLabel:
    """Tests for add_subplot_label."""

    def test_add_label(self):
        fig, ax = plt.subplots()
        add_subplot_label(ax, "a")
        plt.close(fig)

    def test_custom_fontsize(self):
        fig, ax = plt.subplots()
        add_subplot_label(ax, "b", fontsize=16, fontweight="normal")
        plt.close(fig)


class TestSetupAxis:
    """Tests for setup_axis."""

    def test_basic_setup(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="test")
        setup_axis(ax, "X Label", "Y Label", "Title")
        plt.close(fig)

    def test_no_grid(self):
        fig, ax = plt.subplots()
        setup_axis(ax, "X", "Y", "T", grid=False)
        plt.close(fig)

    def test_no_legend(self):
        fig, ax = plt.subplots()
        setup_axis(ax, "X", "Y", "T", legend=False)
        plt.close(fig)

    def test_legend_without_handles(self):
        fig, ax = plt.subplots()
        # No labeled lines - legend should not be added
        setup_axis(ax, "X", "Y", "T", legend=True)
        plt.close(fig)

    def test_custom_legend_loc(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2], label="data")
        setup_axis(ax, "X", "Y", "T", legend_loc="upper left")
        plt.close(fig)


class TestPlotTimeSeries:
    """Tests for plot_time_series."""

    def _make_df(self):
        return pd.DataFrame({
            "time": [0, 1, 2, 3, 4],
            "col_a": [1, 2, 3, 4, 5],
            "col_b": [5, 4, 3, 2, 1],
        })

    def test_basic_plot(self):
        fig, ax = plt.subplots()
        df = self._make_df()
        plot_time_series(ax, df, "time", ["col_a", "col_b"])
        plt.close(fig)

    def test_with_labels(self):
        fig, ax = plt.subplots()
        df = self._make_df()
        plot_time_series(ax, df, "time", ["col_a"], labels=["Series A"])
        plt.close(fig)

    def test_with_colors(self):
        fig, ax = plt.subplots()
        df = self._make_df()
        plot_time_series(ax, df, "time", ["col_a"], colors=["red"])
        plt.close(fig)

    def test_with_linestyles(self):
        fig, ax = plt.subplots()
        df = self._make_df()
        plot_time_series(ax, df, "time", ["col_a"], linestyles=["--"])
        plt.close(fig)

    def test_with_linewidths(self):
        fig, ax = plt.subplots()
        df = self._make_df()
        plot_time_series(ax, df, "time", ["col_a"], linewidths=[3.0])
        plt.close(fig)

    def test_default_linewidth(self):
        fig, ax = plt.subplots()
        df = self._make_df()
        plot_time_series(ax, df, "time", ["col_a", "col_b"])
        plt.close(fig)


class TestAddHorizontalLine:
    """Tests for add_horizontal_line."""

    def test_add_line(self):
        fig, ax = plt.subplots()
        add_horizontal_line(ax, 5.0, "target")
        plt.close(fig)

    def test_custom_style(self):
        fig, ax = plt.subplots()
        add_horizontal_line(ax, 3.0, "ref", color="blue", linestyle="-", alpha=0.8)
        plt.close(fig)


class TestAddTextAnnotation:
    """Tests for add_text_annotation."""

    def test_add_annotation(self):
        fig, ax = plt.subplots()
        add_text_annotation(ax, "Hello")
        plt.close(fig)

    def test_custom_position(self):
        fig, ax = plt.subplots()
        add_text_annotation(ax, "Hi", x=0.5, y=0.5, ha="center", va="center")
        plt.close(fig)


class TestSaveFigure:
    """Tests for save_figure."""

    def test_save(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        output_file = str(tmp_path / "test.png")
        save_figure(fig, output_file)
        assert os.path.exists(output_file)
        plt.close(fig)

    def test_save_custom_dpi(self, tmp_path):
        fig, ax = plt.subplots()
        output_file = str(tmp_path / "test2.png")
        save_figure(fig, output_file, dpi=150, bbox_inches="tight")
        assert os.path.exists(output_file)
        plt.close(fig)


class TestCreateStandardMfcPlots:
    """Tests for create_standard_mfc_plots."""

    def test_create_plots(self, tmp_path):
        df = pd.DataFrame({
            "time_hours": np.linspace(0, 24, 100),
            "reservoir_concentration": np.random.uniform(20, 30, 100),
            "outlet_concentration": np.random.uniform(15, 25, 100),
            "total_power": np.random.uniform(0.01, 0.1, 100),
        })
        prefix = str(tmp_path / "mfc_results")
        fig1, fig2 = create_standard_mfc_plots(df, output_prefix=prefix)
        assert fig1 is not None
        assert fig2 is not None
        plt.close(fig1)
        plt.close(fig2)
