import numpy as np
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.icephys import (
    CurrentClampStimulusSeries,
    CurrentClampSeries,
    VoltageClampSeries,
    VoltageClampStimulusSeries,
    IntracellularElectrode,
)
from uuid import uuid4
from ephys.classes.experiment_objects import ExpData, MetaData, SubjectInfo, DeviceInfo
from ephys.classes.trace import Trace
from ephys.classes.channels import ChannelInformation, Channel
from ephys.classes.current import CurrentClamp, CurrentTrace
from ephys.classes.voltage import VoltageClamp, VoltageTrace

from quantities import Quantity


class NWBExporter:

    def __init__(
        self,
        experiment: ExpData,
        lab="Unknown Lab",
        institution: str = "Unknown Institution",
        session_description: str = "Unknown Session",
        device: DeviceInfo | None = None,
    ) -> None:
        self.experiment = experiment
        self.lab = lab
        self.institution = institution
        self.session_description = session_description

        date_of_exp = self.experiment.meta_data.get_date_of_experiment()
        start_index = np.argmin(date_of_exp)
        self.experimenter = list(set(self.experiment.meta_data.get_experimenter()))
        self.electrode = list()
        self.electrode_groups = list()
        self.response = list()
        self.stimulus = list()

        self.nwb = NWBFile(
            session_description="Ephys recording",
            identifier=str(uuid4()),
            session_start_time=date_of_exp[start_index],
            experimenter=self.experimenter,
            lab=self.lab,
            institution=self.institution,
        )
        self.device = device or self.nwb.create_device(
            name=device.device_name if device else "unknown",
            description=device.device_description if device else "unknown device",
        )

    def add_all_traces(self) -> None:
        for trace in self.experiment.protocols:
            # TODO: check for same electrodes
            self.add_trace(trace)

    def add_trace(self, trace: Trace, location: str = "") -> None:
        """
        Adds a trace to the NWB file.
        """
        for channel_group in np.unique(trace.channel_information.channel_grouping):
            self.add_VC_group(trace, location, channel_grouping=channel_group)

    def add_electrode(
        self, location: str = "", electrode_number: int | None = None
    ) -> IntracellularElectrode:
        electrode = self.nwb.create_icephys_electrode(
            name=(
                f"Electrode_{electrode_number}"
                if electrode_number is not None
                else "Electrode"
            ),
            description="Icephys electrode",
            location=location,
            device=self.device,
        )
        self.electrode.append(electrode)
        return electrode

    def _parse_sweep_data(
        self,
        channel_data: VoltageClamp | CurrentClamp | VoltageTrace | CurrentTrace,
        time: Quantity,
        electrode: IntracellularElectrode,
        sweep_number: int,
        channel_description: str = "",
    ) -> dict:
        return {
            "data": channel_data.data[sweep_number],
            #      "unit": channel_data.unit,
            "timestamps": time.magnitude[sweep_number],
            #          "time_unit": time.dimensionality.string,
            "electrode": electrode,
            #      "rate": float(channel_data.sampling_rate.magnitude),
            "gain": 1.0,
            "resolution": 1.0,
            "conversion": 1.0,
            "description": channel_description,
            "sweep_number": sweep_number,
        }

    def add_VC_group(
        self,
        trace: Trace,
        location: str = "unknown location",
        channel_grouping: int | None = None,
    ) -> None:

        group_indices = np.where(
            trace.channel_information.channel_grouping == channel_grouping
        )[0]
        electrode = self.add_electrode(location=location)
        for channel_index in group_indices:
            channel: Channel = trace.channel[channel_index]
            self.add_channel(
                channel=channel,
                time=trace.time,
                electrode=electrode,
            )

    def add_channel(
        self,
        channel: Channel,
        time: Quantity,
        electrode: IntracellularElectrode,
        channel_description: str = "",
    ) -> None:
        for sweep in range(channel.sweep_count):
            if isinstance(channel, CurrentClamp):
                self.add_current_stimulus_sweep(
                    channel_data=channel,
                    time=time,
                    electrode=electrode,
                    sweep_number=sweep,
                    channel_description=channel_description,
                )
            elif isinstance(channel, VoltageClamp):
                self.add_voltage_stimulus_sweep(
                    channel_data=channel,
                    time=time,
                    electrode=electrode,
                    sweep_number=sweep,
                    channel_description=channel_description,
                )
            elif isinstance(channel, CurrentTrace):
                self.add_current_response_sweep(
                    channel_data=channel,
                    time=time,
                    electrode=electrode,
                    sweep_number=sweep,
                    channel_description=channel_description,
                )
            elif isinstance(channel, VoltageTrace):
                self.add_voltage_response_sweep(
                    channel_data=channel,
                    time=time,
                    electrode=electrode,
                    sweep_number=sweep,
                    channel_description=channel_description,
                )

    def add_current_stimulus_sweep(
        self,
        channel_data: CurrentClamp,
        time: Quantity,
        electrode: IntracellularElectrode,
        sweep_number: int,
        channel_description: str = "Current clamp stimulus",
    ) -> CurrentClampStimulusSeries:
        ex_stim = CurrentClampStimulusSeries(
            name=f"CurrentClampStimulus_ch_{channel_data.channel_number}_sweep_{sweep_number}",
            **self._parse_sweep_data(
                channel_data, time, electrode, sweep_number, channel_description
            ),
        )
        self.nwb.add_stimulus(stimulus=ex_stim)
        return ex_stim

    def add_voltage_stimulus_sweep(
        self,
        channel_data: VoltageClamp,
        time: Quantity,
        electrode: IntracellularElectrode,
        sweep_number: int,
        channel_description: str = "Voltage clamp stimulus",
    ) -> VoltageClampStimulusSeries:
        ex_stim = VoltageClampStimulusSeries(
            name=f"VoltageClampStimulus_ch_{channel_data.channel_number}_sweep_{sweep_number}",
            **self._parse_sweep_data(
                channel_data, time, electrode, sweep_number, channel_description
            ),
        )
        self.nwb.add_stimulus(stimulus=ex_stim)
        return ex_stim

    def add_voltage_response_sweep(
        self,
        channel_data: VoltageTrace,
        time: Quantity,
        electrode: IntracellularElectrode,
        sweep_number: int,
        channel_description: str = "Current clamp response",
    ) -> VoltageClampSeries:
        ex_record = VoltageClampSeries(
            name=f"VoltageClampRecording_ch_{channel_data.channel_number}_sweep_{sweep_number}",
            **self._parse_sweep_data(
                channel_data, time, electrode, sweep_number, channel_description
            ),
        )
        self.nwb.add_acquisition(nwbdata=ex_record)
        return ex_record

    def add_current_response_sweep(
        self,
        channel_data: CurrentTrace,
        time: Quantity,
        electrode: IntracellularElectrode,
        sweep_number: int,
        channel_description: str = "Current clamp response",
    ) -> CurrentClampSeries:
        ex_record = CurrentClampSeries(
            name=f"CurrentClampRecording_ch_{channel_data.channel_number}_sweep_{sweep_number}",
            **self._parse_sweep_data(
                channel_data, time, electrode, sweep_number, channel_description
            ),
        )
        self.nwb.add_acquisition(nwbdata=ex_record)
        return ex_record

    def write_nwb(self, file_path: str) -> None:
        if file_path.endswith(".nwb"):
            with NWBHDF5IO(file_path, "w") as io:
                io.write(self.nwb)
            io.close()
