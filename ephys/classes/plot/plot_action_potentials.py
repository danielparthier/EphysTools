from typing import Any
from ephys.classes.plot.plot_params import PlotParams, _set_axs_color
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from ephys.classes.action_potentials import ActionPotentials
import numpy as np
import pyqtgraph as pg


class ActionPotentialsPlot:
    """
    A class to handle plotting of action potentials.
    """

    def __init__(self, action_potentials: ActionPotentials, **kwargs: Any) -> None:
        """Initialize the ActionPotentialsPlot with action potentials data."""
        if not isinstance(action_potentials, ActionPotentials):
            raise TypeError("action_potentials must be an instance of ActionPotentials")

        self.params = PlotParams(**kwargs)
        self.params.validate()

        self.action_potentials = action_potentials
        self.sweep_numbers = action_potentials.sweep_numbers
        self.time = action_potentials.time
        self.channel = action_potentials.channel
        self.threshold = action_potentials.threshold


class ActionPotentialsMatplotlib(ActionPotentialsPlot):
    """
    A class to handle plotting of action potentials using matplotlib.
    """

    def __init__(self, action_potentials: ActionPotentials):
        super().__init__(action_potentials)

    def plot_matplotlib_action_potentials(
        self,
        sweep_numbers: np.ndarray | list[int] | int | None = None,
        align_threshold: bool = True,
        threshold: bool = False,
        save_path: str = "action_potentials.png",
        show: bool = True,
        save: bool = False,
    ) -> None:
        """
        Plot the action potentials using matplotlib.

        Parameters
        ----------
        sweep_numbers : np.ndarray or list of int, optional
            The sweep numbers to plot. If None, all sweeps are plotted.
        align_threshold : bool, optional
            Whether to align the action potentials to the threshold. Default is True.
        threshold : bool, optional
            Whether to plot the threshold line. Default is False.
        save_path : str, optional
            The path to save the plot. Default is "action_potentials.png".
        show : bool, optional
            Whether to show the plot. Default is True.
        save : bool, optional
            Whether to save the plot. Default is False.
        """

        if len(self.action_potentials.action_potentials) == 0:
            print("No action potentials detected.")
            return None

        if len(self.action_potentials.sweep_numbers) == 0:
            print("No sweep numbers available.")
            return None
        if sweep_numbers is None:
            sweep_numbers = np.unique(self.action_potentials.sweep_numbers)
        else:
            if isinstance(sweep_numbers, int):
                if sweep_numbers not in self.action_potentials.sweep_numbers:
                    print("Sweep number out of range")
                    return None
                sweep_numbers = np.array([sweep_numbers])
            elif isinstance(sweep_numbers, list):
                sweep_numbers = np.array(sweep_numbers)
                if not np.isin(
                    sweep_numbers, self.action_potentials.sweep_numbers
                ).all():
                    print("Sweep number out of range")
                    return None
            elif not isinstance(sweep_numbers, np.ndarray):
                print("Sweep numbers must be a list or numpy array")
                return None
            if len(sweep_numbers) > len(self.action_potentials.sweep_numbers):
                print("Sweep numbers out of range")
                return None
            sweep_numbers = np.unique(sweep_numbers)
        mask = np.isin(self.action_potentials.sweep_numbers, sweep_numbers)
        # action_potentials per sweep
        unique_sweeps = np.unique(self.action_potentials.sweep_numbers[mask])

        # make a plot for each channel
        channels = np.unique(self.action_potentials.channel[mask])
        fig, axs = plt.subplots(
            len(channels), 1, figsize=(8, 4 * len(channels)), squeeze=False
        )
        axs = axs.flatten()

        self._set_axs_color(input_axs=axs)

        fig.set_facecolor(self.params.bg_color)
        for idx, ch in enumerate(channels):
            ch_mask = mask & (self.action_potentials.channel == ch)
            ch_action_potentials = self.action_potentials.action_potentials[ch_mask]
            ch_sweeps = self.action_potentials.sweep_numbers[ch_mask]
            ch_threshold_index = self.action_potentials.threshold["index"][ch_mask]
            for i, action_potential in enumerate(ch_action_potentials):
                sweep_id = ch_sweeps[i]
                sweep_color = cm.get_cmap("gist_rainbow")(
                    (sweep_id - unique_sweeps.min())
                    / (np.ptp(unique_sweeps) if np.ptp(unique_sweeps) else 1)
                )
                if align_threshold:
                    time_scale = (
                        self.action_potentials.time.magnitude
                        - self.action_potentials.time.magnitude[ch_threshold_index[i]]
                    )
                else:
                    time_scale = self.action_potentials.time.magnitude
                axs[idx].plot(
                    time_scale, action_potential, color=sweep_color, alpha=0.2
                )
            axs[idx].set_title(f"Action Potentials - Channel {ch}")
            axs[idx].set_xlabel(
                f"Time ({self.action_potentials.time.dimensionality.string})"
            )
            axs[idx].set_ylabel(
                f"Amplitude ({self.action_potentials.action_potentials.dimensionality.string})"
            )
        if threshold:
            for idx, ch in enumerate(channels):
                ch_mask = mask & (self.action_potentials.channel == ch)
                ch_threshold = self.action_potentials.threshold["threshold"][ch_mask]
                ch_threshold_index = self.action_potentials.threshold["index"][ch_mask]
                ch_sweeps = self.action_potentials.sweep_numbers[ch_mask]
                for i, i_threshold in enumerate(ch_threshold):
                    sweep_id = ch_sweeps[i]
                    sweep_color = cm.get_cmap("gist_rainbow")(
                        (sweep_id - unique_sweeps.min())
                        / (np.ptp(unique_sweeps) if np.ptp(unique_sweeps) else 1)
                    )
                    if align_threshold:
                        time_scale = 0.0
                    else:
                        time_scale = self.action_potentials.time.magnitude[
                            ch_threshold_index[i]
                        ]
                    axs[idx].plot(
                        time_scale,
                        i_threshold,
                        color=sweep_color,
                        marker=".",
                        markersize=5,
                    )

        fig.tight_layout()

        if show:
            plt.show()
        if save:
            plt.savefig(save_path)
        plt.close(fig)
        return None

    def _set_axs_color(self, input_axs: Axes | np.ndarray) -> None:
        _set_axs_color(params=self.params, input_axs=input_axs)


class ActionPotentialsPyQt(ActionPotentialsPlot):
    def __init__(self, action_potentials: ActionPotentials, **kwargs: Any) -> None:
        super().__init__(action_potentials, **kwargs)

    def plot(
        self,
        show: bool = True,
        save: bool = False,
        save_path: str = "action_potentials.png",
    ) -> None:

        self.win = pg.GraphicsLayoutWidget(show=self.params.show, title="Trace Plot")
        self.win.setBackground(self.params.bg_color)
