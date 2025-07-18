from math import e
import sys
import os
import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox,
    QFileDialog, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import gridspec
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import imageio
import Analysis_code_functions as acf
import calibration_calcs as cc
import fine_tuned_espec_analysis as ftes
# Use imageio v2 to avoid deprecation and handle file locks
from imageio import v2 as iio

def parse_burnt_ranges(txt):
    """
    Convert a string of energy ranges into a list of (low, high) tuples. Only parts containing a '-' will be converted, and extra spaces are removed.

    Parameters:
    ----------
        txt (str): A string with energy ranges formatted as "low-high" and separated by commas,
                   e.g., "10-20, 50-60", where each range represents energy bounds in MeV.

    Returns:
    -------
        list[tuple[float, float]]: A list of tuples, each containing the lower and upper energy bounds.
    """
    ranges = []                                     # Start with an empty list to store the energy ranges.
    try:
        if not isinstance(txt, str):
            print(f"Invalid burnt range input: not a string ({txt})")
            return []
        for part in txt.split(','):
            part = part.strip()                         # Remove any extra space around each range.
            if '-' in part:                             # Only process if a '-' is found, indicating a range.
                lo, hi = part.split('-', 1)             # Split into low and high values.
                ranges.append((float(lo), float(hi)))   # Convert strings to floats and append as a tuple.
    except Exception as e:
        print(f"Invalid burnt range: {txt} ({e})")
        ranges = []
    return ranges

def zero_out_ranges(spec, axis, ranges):
    """
    Set spectrum values to zero over specified energy intervals.

    Parameters
    ----------
    spec : np.ndarray
        A 2D array representing the spectral data.
    axis : np.ndarray
        A 1D array of energy values (in MeV) corresponding to spec's columns.
    ranges : list[tuple[float, float]]
        Energy intervals to be masked out, given as (low, high) tuples.

    Returns
    -------
    np.ndarray
        A copy of the spectrum with values set to zero for the specified energy ranges.

    Notes
    -----
    This function is useful for eliminating unwanted energy regions (e.g., due to burnt pixels)
    in the spectrum, so that subsequent analyses are not affected.
    """
    out = spec.copy()                       # Create a copy to preserve the original data.
    for lo, hi in ranges:
        mask = (axis >= lo) & (axis <= hi)  # Identify spectrum positions within the energy range.
        out[mask] = 0                       # Zero out the selected energy bins.
    return out

def process_image_v2(img: np.ndarray, calib: dict, ROI_selection_percent: float = 0.65, 
                     fine_cut_flag: bool = False, bg_flag: bool = True, PT_flag: bool = True, 
                     norm_flag: bool = False, emin: float = None, emax: float = None,
                     burn_ranges: list[tuple[float,float]] = None) -> tuple[np.ndarray,  # post-PT image (2D)
                                                                            np.ndarray,  # deconv & BG-subtracted spectrum image (2D)
                                                                            np.ndarray,  # full energy axis (1D)
                                                                            np.ndarray,  # full 1D energy spectrum
                                                                            np.ndarray,  # divergence per energy bin (1D)
                                                                            np.ndarray,  # masked energy axis (1D)
                                                                            np.ndarray   # masked 1D energy spectrum
                                                                        ]:
    """
    v2 wrapper for ESPEC live GUI. 
    Takes a raw image and calibration dict, returns:
      - post_PT_image           : 2D array for panel 1
      - spectrum_img_corr       : 2D deconvolved & BG-subtracted image for panel 2
      - energy_axis             : 1D array of energy (MeV) for panel 2 x-axis
      - intensity_distribution  : 1D energy spectrum for panel 3
      - divergence_vals         : 1D divergence array for panel 3
      - masked_energy_axis      : 1D masked energy axis (if burn_ranges provided)
      - masked_intensity        : 1D masked intensity spectrum (if burn_ranges provided)

    Calibration dict must contain keys:
      maxWid, maxLen, projmat, pixel2mm, screen_to_axis_distance,
      dE_ds_interp, average_distance_covered, energy_interp, energy_partition
    """
    # ensure float32
    trial = img.astype(np.float32)

    # perspective transform
    if PT_flag:
        post_PT = acf.unwarp_perspective(
            trial, calib['projmat'], (calib['maxWid'], calib['maxLen'])
        )
    else:
        post_PT = trial

    # saving a copy of the post-PT image
    raw_PT = post_PT.copy()

    # build screen_mm axis
    h, w = post_PT.shape
    screen_mm = np.linspace(
        calib['screen_to_axis_distance'],
        calib['screen_to_axis_distance'] + w * calib['pixel2mm'],
        w    )

    print(f"Screen distance axis (mm): max={max(screen_mm)}, min={min(screen_mm)}")
    # mm → energy
    energy_axis = calib['energy_interp'](screen_mm)

    # background subtraction & divergence
    post_bg, divergence_vals, energy_sections = acf.cuts_and_BG(
        post_PT,
        energy_axis,
        x_axis=energy_axis,
        y_axis=np.arange(h),
        pixel2mm=calib['pixel2mm'],
        average_distance_covered=calib['average_distance_covered'],
        fine_cut=fine_cut_flag,
        bg_sub=bg_flag,
        energy_partition=calib['energy_partition'],
        selection_area_percentage=ROI_selection_percent
    )

    # charge-conserving conversion
    dE_ds = calib['dE_ds_interp'](screen_mm) * -1
    # Divide each pixel in a column of post_bg with the respective dE_ds values to conserve charge on screen
    spec2D = post_bg / dE_ds

    # integrate → 1D spectrum
    full_spec1D = spec2D.sum(axis=0)

    # optional normalization
    if norm_flag and full_spec1D.max() > 0:
        full_spec1D = full_spec1D / full_spec1D.max()

    # build mask for emin/emax & burn_ranges
    mask = np.ones_like(energy_axis, dtype=bool)
    if emin is not None:
        mask &= (energy_axis >= emin)
    if emax is not None:
        mask &= (energy_axis <= emax)
    if burn_ranges:
        for lo, hi in burn_ranges:
            mask &= ~((energy_axis >= lo) & (energy_axis <= hi))

    masked_energy = energy_axis[mask]
    masked_spec   = full_spec1D[mask]

    # divergence axis
    divergence_axis = np.array(energy_sections)

    return raw_PT, spec2D, energy_axis, full_spec1D, divergence_vals, divergence_axis, masked_energy, masked_spec

class GuiSignals(QObject):
    log_message = pyqtSignal(str)
    new_file = pyqtSignal(str)

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, gui):
        self.gui = gui
        self.signals = GuiSignals()
        self.signals.log_message.connect(gui.log.append)
        self.signals.new_file.connect(gui.handle_new_file)

    def on_created(self, event):
        # Triggered when a new file appears in the watched folder
        if event.is_directory:
            return
        path = event.src_path
        if not path.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
            return
        filename = os.path.basename(path)
        self.signals.log_message.emit(f"[INFO] New file detected: {filename}")
        self.signals.new_file.emit(path)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # make the figure taller
        fig = Figure(figsize=(14, 16))    # ↑ height up from 10→12
        super().__init__(fig)
        self.setParent(parent)

        # Grid: 3 rows, 2 cols
        #  • width_ratios unchanged
        #  • height_ratios=[3,3,2] gives more space to the top two rows
        #  • hspace=0.5 increases vertical padding between them    
        gs = gridspec.GridSpec(
            3, 2,
            width_ratios=[24, 1],
            height_ratios=[3, 3, 3],   # ↑ give relative row‐heights
            wspace=0.15,
            hspace=0.6                 # ↑ more vertical gap
        )
        self.ax_img   = fig.add_subplot(gs[0, 0])
        self.cax_img  = fig.add_subplot(gs[0, 1])
        self.ax_spec  = fig.add_subplot(gs[1, 0])
        self.cax_spec = fig.add_subplot(gs[1, 1])
        self.ax_prof  = fig.add_subplot(gs[2, 0])
        # — lower-right: will become our vertical score gauge —
        self.cax_score = fig.add_subplot(gs[2, 1])
        # start off empty
        self.cax_score.axis('off')

    def update_plots(self,
                     post_PT: np.ndarray = None,
                     spec2D: np.ndarray  = None,
                     energy_axis: np.ndarray = None,
                     masked_energy: np.ndarray = None,
                     masked_intensity: np.ndarray = None,
                     pixel2mm: float = 1.0,
                     score_dict: dict = None,
                     target_peak: float = None,
                     target_sigma:float = None):
        """
        Draw three panels:
          1) post_PT image + slim 'Counts' colorbar
          2) spec2D image + slim 'Counts' colorbar
          3) masked lineout (no colorbar)
        pixel2mm: conversion factor for y-axes in mm
        """
        # Clear axes
        for ax in (self.ax_img, self.cax_img,
                   self.ax_spec, self.cax_spec,
                   self.ax_prof):
            ax.clear()

        cmap = 'viridis'

        # Panel 1: Post-PT Image
        if post_PT is not None:
            im = self.ax_img.imshow(
                post_PT, aspect='auto', origin='lower', cmap=cmap
            )
            # X-axis: columns → mm
            w = post_PT.shape[1]
            xt = np.linspace(0, w, 6)
            self.ax_img.set_xticks(xt)
            self.ax_img.set_xticklabels((xt * pixel2mm).astype(int))
            # Y-axis: rows → mm
            h = post_PT.shape[0]
            yt = np.linspace(0, h, 6)
            self.ax_img.set_yticks(yt)
            self.ax_img.set_yticklabels((yt * pixel2mm).astype(int))
            self.ax_img.set_xlabel('Screen Distance (mm)')
            self.ax_img.set_ylabel('Screen Position (mm)')
            cb1 = self.ax_img.figure.colorbar(
                im, cax=self.cax_img, shrink=0.5, label='Counts'
            )
            cb1.outline.set_visible(False)
        else:
            self.cax_img.axis('off')

        # Panel 2: 2D Spectrum Image
        if spec2D is not None and energy_axis is not None:
            y = np.arange(spec2D.shape[0])
            m2 = self.ax_spec.pcolormesh(
                energy_axis, y, spec2D, cmap=cmap, shading='auto'
            )
            # Y-axis in mm
            yt = np.linspace(0, spec2D.shape[0], 6)
            self.ax_spec.set_yticks(yt)
            self.ax_spec.set_yticklabels((yt * pixel2mm).astype(int))
            self.ax_spec.set_xlabel('Energy (MeV)')
            self.ax_spec.set_ylabel('Screen Position (mm)')
            cb2 = self.ax_spec.figure.colorbar(
                m2, cax=self.cax_spec, shrink=0.5, label='Counts'
            )
            cb2.outline.set_visible(False)
        else:
            self.cax_spec.axis('off')

        # Panel 3: Masked Lineout
        if masked_energy is not None and masked_intensity is not None:
            self.ax_prof.plot(masked_energy, masked_intensity, lw=1, label='Experimental')

        # build the target shape if the target params were passed
        if (masked_energy is not None and
            masked_intensity is not None and
            target_peak is not None and
            target_sigma is not None):

            raw_target = ftes.aimed_spectrum(
                masked_energy,
                peak=target_peak,
                sigma=target_sigma
            )
            # Ensure scaling is robust even if masked_intensity is flat or zero
            exp_max = np.nanmax(masked_intensity) if masked_intensity.size else 1.0
            tar_max = np.nanmax(raw_target) if raw_target.size else 1.0
            scale = exp_max / tar_max if tar_max > 0 else 1.0
            # If exp_max is zero, show the target at a fixed height (e.g. 1)
            if exp_max == 0:
                scale = 1.0
            scaled_target = raw_target * scale

            self.ax_prof.plot(
                masked_energy, scaled_target,
                lw=2, color='red', linestyle='--',
                label='Target (scaled)'
            )

        self.ax_prof.legend(loc='upper right')
        self.ax_prof.set_xlabel('Energy (MeV)')
        self.ax_prof.set_ylabel('Counts')
        self.ax_prof.set_title('Masked Lineout')
        self.ax_prof.grid(True)

        # Panel 4: Vertical 0–100 score gauge
        self.cax_score.clear()
        if score_dict is not None:
            score = score_dict['score']
            # 1) full axis range 0–100
            self.cax_score.set_xlim(0, 1)
            self.cax_score.set_ylim(0, 100)
            # 2) draw a single vertical bar at x=0.5
            self.cax_score.bar(0.5, score, width=0.6)
            # 3) clean up spines & ticks
            for spine in self.cax_score.spines.values():
                spine.set_visible(False)
            self.cax_score.get_xaxis().set_visible(False)
            # Move the y ticks and label to the left side
            self.cax_score.yaxis.tick_right()
            self.cax_score.yaxis.set_label_position('right')
            # 4) label the axis and annotate the numeric value
            self.cax_score.set_ylabel('Score', rotation = 90, labelpad=8)
            self.cax_score.text(
                0.5, score + 3,
                f"{score:.1f}",
                ha='center', va='bottom',
                transform=self.cax_score.transAxes
            )
        else:
            # if no score yet, just hide everything
            self.cax_score.axis('off')

        self.draw()  # Redraw the canvas to update the plots

class MainWindow(QMainWindow):
    def handle_new_file(self, path):
        # This runs in the main Qt thread
        if self.calib is None:
            return
        # Wait for file to be fully written
        max_attempts = 50
        last_size = -1
        stable_count = 0
        time.sleep(0.05)
        for attempt in range(max_attempts):
            try:
                size = os.path.getsize(path)
                if size == last_size and size > 0:
                    stable_count += 1
                else:
                    stable_count = 0
                last_size = size
                with open(path, 'rb'):
                    pass
                if stable_count >= 2:
                    img = iio.imread(path)
                    break
            except (PermissionError, OSError) as e:
                self.log.append(f"[INFO] File read attempt {attempt+1} failed: {e}")
            time.sleep(0.05)
        else:
            self.log.append(f"[WARN] Cannot read file (locked or incomplete): {os.path.basename(path)}")
            return
        espec_img = img.astype(np.float32)
        try:
            postPT, spec2D, e_axis, full1D, div_vals, div_axis, m_e, m_spec = process_image_v2(
                img=espec_img,
                calib=self.calib,
                ROI_selection_percent=0.65,
                fine_cut_flag=True,
                bg_flag=True,
                PT_flag=True,
                norm_flag=False,
                emin=self.spin_minE.value(),
                emax=self.spin_maxE.value(),
                burn_ranges=parse_burnt_ranges(self.edit_burnt.text())
            )
            emin, emax = self.spin_minE.value(), self.spin_maxE.value()
            div_mask = (div_axis >= emin) & (div_axis <= emax)
            m_div_axis = div_axis[div_mask]
            m_div_vals = np.array(div_vals)[div_mask]
            if not hasattr(self, 'shot'):
                self.shot = 1
            else:
                self.shot += 1
            exp_spec = np.column_stack((m_e, m_spec))
            target = lambda e: ftes.aimed_spectrum(e, peak=self.spin_target_peak.value(), sigma=self.spin_target_sigma.value())
            score_dict = ftes.evaluate_spectrum_v3(
                exp_spectrum          = exp_spec,
                target_spectrum_func  = target,
                region                = (emin, emax),  
                low_energy_threshold  = 50.0,
                target_peak           = self.spin_target_peak.value(),
                weights = {
                    'shape':      self.spin_w_shape.value(),
                    'peak_pos':   self.spin_w_peak.value(),
                    'spread':     self.spin_w_spread.value(),
                    'low_energy': self.spin_w_lowE.value(),
                    'kurtosis':   self.spin_w_kurt.value(),
                    'multipeak':  self.spin_w_multipeak.value(),
                    'overshoot':  self.spin_w_overshoot.value(),
                },
                worst_cost = self.spin_cost.value()
            )
            self.plot_canvas.update_plots(
                post_PT=postPT,
                spec2D=spec2D,
                energy_axis=e_axis,
                masked_energy=m_e,
                masked_intensity=m_spec,
                pixel2mm=self.calib['pixel2mm'],
                score_dict=score_dict,
                target_peak=self.spin_target_peak.value(),
                target_sigma=self.spin_target_sigma.value()
            )
            self.lbl_score_shape.setText(f"{score_dict['shape']:.2f}")
            self.lbl_score_peak.setText(f"{score_dict['peak']:.2f}")
            self.lbl_score_spread.setText(f"{score_dict['spread']:.2f}")
            self.lbl_score_kurt.setText(f"{score_dict['kurtosis']:.2f}")
            self.lbl_score_multipeak.setText(f"{score_dict['multipeak']:.2f}")
            self.lbl_score_lowE.setText(f"{score_dict['low']:.2f}")
            self.lbl_score_overshoot.setText(f"{score_dict['over']:.2f}")
            self.lbl_score_totalcost.setText(f"{score_dict['raw']:.2f}")
            txt = (f"Shot {self.shot} → Score {score_dict['score']:.2f}  "
                f"(raw cost={score_dict['raw']:.2f})")
            self.log.append(txt)
            if self.output_folder:
                fname = os.path.join(self.output_folder, f"shot_{self.shot}_summary.txt")
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(txt + "\n")
        except Exception as ex:
            self.log.append(f"[ERROR] Processing failed: {ex}")
            import traceback
            self.log.append(traceback.format_exc())
            return
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ESPEC Live Analysis')
        self.resize(1700, 900)
        self.calib = None
        self.input_folder = None
        self.output_folder = None
        self.observer = None

        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout = QVBoxLayout(central)
        layout.addWidget(splitter)

        # Left Control Panel
        control = QWidget()
        control.setFixedWidth(350)  # Fixed sidebar width
        control_layout = QVBoxLayout(control)

        # Calibration Group
        calib_group = QGroupBox('Calibration')
        calib_layout = QVBoxLayout(calib_group)
        self.btn_create_calib = QPushButton('Create Calibration…')
        self.btn_load_calib = QPushButton('Load Calibration File…')
        calib_layout.addWidget(self.btn_create_calib)
        calib_layout.addWidget(self.btn_load_calib)
        self.btn_create_calib.clicked.connect(self.create_calibration)
        self.btn_load_calib.clicked.connect(self.load_calibration)
        control_layout.addWidget(calib_group)

        # Folders Group
        folders_group = QGroupBox('Folders')
        folders_layout = QVBoxLayout(folders_group)
        in_layout = QHBoxLayout()
        in_layout.addWidget(QLabel('Input:'))
        self.lbl_in = QLabel('None')
        self.lbl_in.setMinimumWidth(120)
        self.lbl_in.setMaximumWidth(180)
        self.lbl_in.setTextInteractionFlags(Qt.TextSelectableByMouse)
        in_layout.addWidget(self.lbl_in)
        self.btn_in = QPushButton('Select Input…')
        in_layout.addWidget(self.btn_in)
        folders_layout.addLayout(in_layout)
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel('Output:'))
        self.lbl_out = QLabel('None')
        self.lbl_out.setMinimumWidth(120)
        self.lbl_out.setMaximumWidth(180)
        self.lbl_out.setTextInteractionFlags(Qt.TextSelectableByMouse)
        out_layout.addWidget(self.lbl_out)
        self.btn_out = QPushButton('Select Output…')
        out_layout.addWidget(self.btn_out)
        folders_layout.addLayout(out_layout)
        control_layout.addWidget(folders_group)

        # Analysis Parameters
        params_group = QGroupBox('Analysis Parameters')
        params_layout = QVBoxLayout(params_group)
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel('Min E (MeV):'))
        self.spin_minE = QDoubleSpinBox(); self.spin_minE.setRange(0,1000)
        energy_layout.addWidget(self.spin_minE)
        energy_layout.addWidget(QLabel('Max E (MeV):'))
        self.spin_maxE = QDoubleSpinBox(); self.spin_maxE.setRange(0,1000); self.spin_maxE.setValue(200)
        energy_layout.addWidget(self.spin_maxE)
        params_layout.addLayout(energy_layout)
        params_layout.addWidget(QLabel('Burnt-pixel ranges:'))
        self.edit_burnt = QLineEdit()
        params_layout.addWidget(self.edit_burnt)
        control_layout.addWidget(params_group)

        # Target Spectrum
        target_group = QGroupBox('Target Spectrum')
        target_layout = QFormLayout(target_group)
        self.spin_target_peak = QDoubleSpinBox(); self.spin_target_peak.setRange(0,1000); self.spin_target_peak.setValue(100)
        target_layout.addRow('Target Peak (MeV):', self.spin_target_peak)
        self.spin_target_sigma = QDoubleSpinBox(); self.spin_target_sigma.setRange(0,100); self.spin_target_sigma.setValue(5)
        target_layout.addRow('Target Sigma (MeV):', self.spin_target_sigma)
        control_layout.addWidget(target_group)

        # Scoring Parameters
        scoring_group = QGroupBox('Scoring Parameters')
        scoring_layout = QFormLayout(scoring_group)

        # Worst‐case cost
        self.spin_cost = QDoubleSpinBox()
        self.spin_cost.setRange(-1e1, 1e3)
        self.spin_cost.setValue(70.0)
        self.spin_cost.setSingleStep(0.1)
        # scoring_layout.addRow('Worst Cost:', self.spin_cost)
        # — Worst Cost + live raw cost display —
        widget_cost = QWidget()
        h_cost = QHBoxLayout(widget_cost)
        h_cost.setContentsMargins(0,0,0,0)
        h_cost.addWidget(self.spin_cost)
        #### Weights for shape scoring
        self.spin_w_shape = QDoubleSpinBox()
        self.spin_w_shape.setRange(-1e2, 1e3)
        self.spin_w_shape.setSingleStep(0.1)
        self.spin_w_shape.setValue(35.0)               # default
        # scoring_layout.addRow('Shape Weight:', self.spin_w_shape)
        # — Shape Weight + live score display —
        widget_w_shape = QWidget()
        h_w_shape = QHBoxLayout(widget_w_shape)
        h_w_shape.setContentsMargins(0,0,0,0)
        h_w_shape.addWidget(self.spin_w_shape)
        self.lbl_score_shape = QLabel("0.00")
        h_w_shape.addWidget(self.lbl_score_shape)
        scoring_layout.addRow("Shape Weight:", widget_w_shape)

        #### Peak‐position weight
        self.spin_w_peak = QDoubleSpinBox()
        self.spin_w_peak.setRange(-1e2, 1e3)
        self.spin_w_peak.setSingleStep(0.1)
        self.spin_w_peak.setValue(15.0)                # default 
        # scoring_layout.addRow('Peak Pos Weight:', self.spin_w_peak)
        # — Peak Pos Weight + live score display —
        widget_w_peak = QWidget()
        h_w_peak = QHBoxLayout(widget_w_peak)
        h_w_peak.setContentsMargins(0,0,0,0)
        h_w_peak.addWidget(self.spin_w_peak)
        self.lbl_score_peak = QLabel("0.00")
        h_w_peak.addWidget(self.lbl_score_peak)
        scoring_layout.addRow("Peak Pos Weight:", widget_w_peak)
                
        #### Weights for Spread scoring
        self.spin_w_spread = QDoubleSpinBox(); 
        self.spin_w_spread.setRange(-1e2, 1e3); 
        self.spin_w_spread.setSingleStep(0.1)
        self.spin_w_spread.setValue(5.0)               # default
        # scoring_layout.addRow('Spread Weight:', self.spin_w_spread)
        # — Spread Weight + live score display —
        widget_w_spread = QWidget()
        h_w_spread = QHBoxLayout(widget_w_spread)
        h_w_spread.setContentsMargins(0,0,0,0)
        h_w_spread.addWidget(self.spin_w_spread)
        self.lbl_score_spread = QLabel("0.00")
        h_w_spread.addWidget(self.lbl_score_spread)
        scoring_layout.addRow("Spread Weight:", widget_w_spread)

        #### Weights for Kurtosis scoring
        self.spin_w_kurt = QDoubleSpinBox(); 
        self.spin_w_kurt.setRange(-1e2, 1e3); 
        self.spin_w_kurt.setSingleStep(0.1)
        self.spin_w_kurt.setValue(0.1)                 # default
        # scoring_layout.addRow('Kurtosis Weight:', self.spin_w_kurt)
        # — Kurtosis Weight + live score display —
        widget_w_kurt = QWidget()
        h_w_kurt = QHBoxLayout(widget_w_kurt)
        h_w_kurt.setContentsMargins(0,0,0,0)
        h_w_kurt.addWidget(self.spin_w_kurt)
        self.lbl_score_kurt = QLabel("0.00")
        h_w_kurt.addWidget(self.lbl_score_kurt)
        scoring_layout.addRow("Kurtosis Weight:", widget_w_kurt)

        #### Weights for Multi-peak scoring
        self.spin_w_multipeak = QDoubleSpinBox(); 
        self.spin_w_multipeak.setRange(-1e2, 1e3); 
        self.spin_w_multipeak.setSingleStep(0.1)
        self.spin_w_multipeak.setValue(30.0)            # default
        # scoring_layout.addRow('Multi-peak Weight:', self.spin_w_multipeak)
        # — Multi-peak Weight + live score display —
        widget_w_multipeak = QWidget()
        h_w_multipeak = QHBoxLayout(widget_w_multipeak)
        h_w_multipeak.setContentsMargins(0,0,0,0)
        h_w_multipeak.addWidget(self.spin_w_multipeak)
        self.lbl_score_multipeak = QLabel("0.00")
        h_w_multipeak.addWidget(self.lbl_score_multipeak)
        scoring_layout.addRow("Multi-peak Weight:", widget_w_multipeak)
        
        #### Weights for Low-E penalty
        self.spin_w_lowE = QDoubleSpinBox(); 
        self.spin_w_lowE.setRange(-1e2, 1e3); 
        self.spin_w_lowE.setSingleStep(0.1)
        self.spin_w_lowE.setValue(-1.5)                # default
        # scoring_layout.addRow('Low-E Penalty Weight:', self.spin_w_lowE)
        # — Low-E Penalty Weight + live score display —
        widget_w_lowE = QWidget()
        h_w_lowE = QHBoxLayout(widget_w_lowE)
        h_w_lowE.setContentsMargins(0,0,0,0)
        h_w_lowE.addWidget(self.spin_w_lowE)
        self.lbl_score_lowE = QLabel("0.00")
        h_w_lowE.addWidget(self.lbl_score_lowE)
        scoring_layout.addRow("Low-E Penalty Weight:", widget_w_lowE)

        #### Weights for Overshoot bonus
        self.spin_w_overshoot = QDoubleSpinBox(); 
        self.spin_w_overshoot.setRange(-1e2, 1e3); 
        self.spin_w_overshoot.setSingleStep(0.1)
        self.spin_w_overshoot.setValue(1.5)             # default
        # scoring_layout.addRow('Overshoot Bonus Weight:', self.spin_w_overshoot)
        # — Overshoot Bonus Weight + live score display —
        widget_w_overshoot = QWidget()
        h_w_overshoot = QHBoxLayout(widget_w_overshoot)
        h_w_overshoot.setContentsMargins(0,0,0,0)
        h_w_overshoot.addWidget(self.spin_w_overshoot)
        self.lbl_score_overshoot = QLabel("0.00")
        h_w_overshoot.addWidget(self.lbl_score_overshoot)
        scoring_layout.addRow("Overshoot Bonus Weight:", widget_w_overshoot)

        control_layout.addWidget(scoring_group)

        # Activity Log
        control_layout.addWidget(QLabel('Activity Log:'))
        self.log = QTextEdit(); self.log.setReadOnly(True)
        control_layout.addWidget(self.log)
        control_layout.addStretch()

        # Right Display Panel
        display = QWidget(); display_layout = QVBoxLayout(display)
        self.plot_canvas = PlotCanvas(self)
        display_layout.addWidget(self.plot_canvas)

        splitter.addWidget(control)
        splitter.addWidget(display)
        splitter.setSizes([350, 850])

        # Connect selectors
        self.btn_in.clicked.connect(self.select_input_folder)
        self.btn_out.clicked.connect(self.select_output_folder)

    def create_calibration(self):
        try:
            cc.save_calibration_data()
        except Exception as ex:
            self.log.append(f"[ERROR] Calibration creation failed: {ex}")

    def load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load Calibration File', filter='*.txt')
        if path:
            try:
                self.calib = ftes.load_and_prepare_calibration(path)
                self.log.append(f'Calibration loaded: {path}')
                self.restart_observer()
            except Exception as ex:
                self.log.append(f"[ERROR] Failed to load calibration: {ex}")

    def select_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if path:
            self.input_folder = path
            self.lbl_in.setText(self._elide_path(path, self.lbl_in))
            self.lbl_in.setToolTip(path)  # Show full path on hover
            self.restart_observer()

    def select_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if path:
            self.output_folder = path
            self.lbl_out.setText(self._elide_path(path, self.lbl_out))
            self.lbl_out.setToolTip(path)  # Show full path on hover
            self.restart_observer()
    def _elide_path(self, path, label):
        """
        Truncate the path with ellipsis if it exceeds the label's max width.
        """
        metrics = label.fontMetrics()
        max_width = label.maximumWidth() if label.maximumWidth() > 0 else 180
        return metrics.elidedText(path, Qt.ElideMiddle, max_width)

    def restart_observer(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        if self.input_folder and self.output_folder and self.calib:
            try:
                handler = NewFileHandler(self)
                # Use PollingObserver for network folder compatibility
                self.observer = PollingObserver()
                self.observer.schedule(handler, self.input_folder, recursive=False)
                self.observer.start()
                self.log.append(f'Watching {self.input_folder}')
            except Exception as ex:
                self.log.append(f"[ERROR] Failed to start observer: {ex}")

    def closeEvent(self, event):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        super().closeEvent(event)

    # Optional: Validate spin box values before processing
    def validate_parameters(self):
        if self.spin_minE.value() >= self.spin_maxE.value():
            self.log.append("[WARN] Min E must be less than Max E.")
            return False
        # Add more checks as needed
        return True

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        w = MainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception as ex:
        print(f"[FATAL] Application crashed: {ex}")
