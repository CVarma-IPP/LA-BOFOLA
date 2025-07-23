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
        w
    )

    print(f"Screen distance axis (mm): max={max(screen_mm)}, min={min(screen_mm)}")
    # mm → energy
    energy_axis = calib['energy_interp'](screen_mm)
    print(f"Energy axis (MeV): max={max(energy_axis)}, min={min(energy_axis)}")

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
        # Use Qt.ConnectionType.QueuedConnection to ensure thread safety
        self.signals.log_message.connect(gui.log.append, Qt.ConnectionType.QueuedConnection)
        self.signals.new_file.connect(gui.handle_new_file, Qt.ConnectionType.QueuedConnection)

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
            # X-axis: columns → mm (use actual screen_mm values)
            w = post_PT.shape[1]
            
            # Calculate screen_mm axis
            if hasattr(self, 'parent') and hasattr(self.parent(), 'calib'):
                calib = self.parent().calib
            else:
                calib = None
            if calib is not None:
                # Use the same formula as in process_image_v2, but ensure axis is in mm (not pixels)
                screen_mm = np.linspace(
                    calib['screen_to_axis_distance'],
                    calib['screen_to_axis_distance'] + w * calib['pixel2mm'],
                    w
                )
                print('[DEBUG] Using calib for axis ticks. pixel2mm:', calib['pixel2mm'])
                xt_idx = np.linspace(0, w-1, 6).astype(int)
                xticks = xt_idx
                xticklabels = [f"{screen_mm[i]:.1f}" for i in xt_idx]
                self.ax_img.set_xticks(xticks)
                self.ax_img.set_xticklabels(xticklabels)
            else:
                print('[DEBUG] Using fallback (pixels) for axis ticks.')
                xt = np.linspace(0, w, 6)
                self.ax_img.set_xticks(xt)
                self.ax_img.set_xticklabels((xt * pixel2mm).astype(int))
            # Y-axis: rows → mm
            h = post_PT.shape[0]
            yt = np.linspace(0, h-1, 6).astype(int)
            yticks = yt
            if calib is not None:
                screen_mm_y = yt * calib['pixel2mm']
                yticklabels = [f"{val:.1f}" for val in screen_mm_y]
                self.ax_img.set_yticks(yticks)
                self.ax_img.set_yticklabels(yticklabels)
            else:
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

class AnalysisWorker(QThread):
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, gui, path):
        super().__init__()
        self.gui = gui
        self.path = path
    
    def run(self):
        import time
        print(f'[DEBUG] AnalysisWorker started for {self.path}')
        try:
            # --- Robust file readiness logic ---
            max_wait = 10.0  # seconds
            min_stable_checks = 3
            backoff = 0.05   # initial sleep (seconds)
            max_backoff = 0.5
            stable_count = 0
            last_size = -1
            start_time = time.time()
            img = None

            while True:
                try:
                    size = os.path.getsize(self.path)
                    if size == last_size and size > 0:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_size = size

                    if stable_count >= min_stable_checks:
                        # Try to read header and image
                        with open(self.path, 'rb') as f:
                            header = f.read(1024)
                            if len(header) < 8:
                                raise ValueError("File too small")
                        test_img = iio.imread(self.path)
                        if test_img is not None and hasattr(test_img, 'size') and test_img.size > 0:
                            img = test_img
                            break
                        else:
                            raise ValueError("Empty or invalid image")
                except (ValueError, OSError, PermissionError) as read_error:
                    stable_count = 0
                except (PermissionError, OSError, FileNotFoundError):
                    pass

                if time.time() - start_time > max_wait:
                    self.error.emit(f"[WARN] Cannot read file after {max_wait:.1f}s (locked, incomplete, or corrupted): {os.path.basename(self.path)}")
                    return
                time.sleep(backoff)
                backoff = min(backoff * 1.5, max_backoff)
            # --- End file readiness logic ---

            espec_img = img.astype(np.float32)
            gui = self.gui
            emin = gui.spin_minE.value()
            emax = gui.spin_maxE.value()
            burnt_ranges = parse_burnt_ranges(gui.edit_burnt.text())
            target_peak = gui.spin_target_peak.value()
            target_sigma = gui.spin_target_sigma.value()
            w_shape = gui.spin_w_shape.value()
            w_peak = gui.spin_w_peak.value()
            w_spread = gui.spin_w_spread.value()
            w_lowE = gui.spin_w_lowE.value()
            w_kurt = gui.spin_w_kurt.value()
            w_multipeak = gui.spin_w_multipeak.value()
            w_overshoot = gui.spin_w_overshoot.value()
            worst_cost = gui.spin_cost.value()
            postPT, spec2D, e_axis, full1D, div_vals, div_axis, m_e, m_spec = process_image_v2(
                img=espec_img,
                calib=gui.calib,
                ROI_selection_percent=0.65,
                fine_cut_flag=True,
                bg_flag=True,
                PT_flag=True,
                norm_flag=False,
                emin=emin,
                emax=emax,
                burn_ranges=burnt_ranges
            )
            div_mask = (div_axis >= emin) & (div_axis <= emax)
            m_div_axis = div_axis[div_mask]
            m_div_vals = np.array(div_vals)[div_mask]
            if not hasattr(gui, 'shot'):
                gui.shot = 1
            else:
                gui.shot += 1
            exp_spec = np.column_stack((m_e, m_spec))
            target = lambda e: ftes.aimed_spectrum(e, peak=target_peak, sigma=target_sigma)
            score_dict = ftes.evaluate_spectrum_v3(
                exp_spectrum          = exp_spec,
                target_spectrum_func  = target,
                region                = (emin, emax),  
                low_energy_threshold  = 50.0,
                target_peak           = target_peak,
                weights = {
                    'shape':      w_shape,
                    'peak_pos':   w_peak,
                    'spread':     w_spread,
                    'low_energy': w_lowE,
                    'kurtosis':   w_kurt,
                    'multipeak':  w_multipeak,
                    'overshoot':  w_overshoot,
                },
                worst_cost = worst_cost
            )
            result = {
                'postPT': postPT,
                'spec2D': spec2D,
                'e_axis': e_axis,
                'm_e': m_e,
                'm_spec': m_spec,
                'score_dict': score_dict,
                'shot': gui.shot
            }
            self.result_ready.emit(result)
            print(f'[DEBUG] AnalysisWorker finished for {self.path}')
        except Exception as ex:
            import traceback
            print(f'[ERROR] AnalysisWorker crashed for {self.path}: {ex}')
            print(traceback.format_exc())
            self.error.emit(f"[ERROR] Processing failed: {ex}\n{traceback.format_exc()}")

class MainWindow(QMainWindow):
       
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ESPEC Live Analysis')
        self.resize(1700, 900)
        self.calib = None
        self.input_folder = None
        self.output_folder = None
        self.observer = None
        self._closing = False  # Flag to track if window is closing
        # self.batch_timer = None
        # self.batch_files = []
        # self.batch_index = 0
        self._batch_workers = []  # Track running batch workers

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
        self.lbl_in.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        in_layout.addWidget(self.lbl_in)
        self.btn_in = QPushButton('Select Input…')
        in_layout.addWidget(self.btn_in)
        print('[DEBUG] Connecting btn_in to select_input_folder...')
        self.btn_in.clicked.connect(self.select_input_folder)
        folders_layout.addLayout(in_layout)
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel('Output:'))
        self.lbl_out = QLabel('None')
        self.lbl_out.setMinimumWidth(120)
        self.lbl_out.setMaximumWidth(180)
        self.lbl_out.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        out_layout.addWidget(self.lbl_out)
        self.btn_out = QPushButton('Select Output…')
        out_layout.addWidget(self.btn_out)
        print('[DEBUG] Connecting btn_out to select_output_folder...')
        self.btn_out.clicked.connect(self.select_output_folder)
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
        widget_cost = QWidget()
        h_cost = QHBoxLayout(widget_cost)
        h_cost.setContentsMargins(0,0,0,0)
        h_cost.addWidget(self.spin_cost)
        self.lbl_score_totalcost = QLabel("0.00")
        h_cost.addWidget(self.lbl_score_totalcost)
        scoring_layout.addRow('Worst Cost:', widget_cost)

        # Shape Weight
        self.spin_w_shape = QDoubleSpinBox()
        self.spin_w_shape.setRange(-1e2, 1e3)
        self.spin_w_shape.setSingleStep(0.1)
        self.spin_w_shape.setValue(35.0)
        widget_w_shape = QWidget()
        h_w_shape = QHBoxLayout(widget_w_shape)
        h_w_shape.setContentsMargins(0,0,0,0)
        h_w_shape.addWidget(self.spin_w_shape)
        self.lbl_score_shape = QLabel("0.00")
        h_w_shape.addWidget(self.lbl_score_shape)
        scoring_layout.addRow("Shape Weight:", widget_w_shape)

        # Peak Position Weight
        self.spin_w_peak = QDoubleSpinBox()
        self.spin_w_peak.setRange(-1e2, 1e3)
        self.spin_w_peak.setSingleStep(0.1)
        self.spin_w_peak.setValue(15.0)
        widget_w_peak = QWidget()
        h_w_peak = QHBoxLayout(widget_w_peak)
        h_w_peak.setContentsMargins(0,0,0,0)
        h_w_peak.addWidget(self.spin_w_peak)
        self.lbl_score_peak = QLabel("0.00")
        h_w_peak.addWidget(self.lbl_score_peak)
        scoring_layout.addRow("Peak Pos Weight:", widget_w_peak)

        # Spread Weight
        self.spin_w_spread = QDoubleSpinBox()
        self.spin_w_spread.setRange(-1e2, 1e3)
        self.spin_w_spread.setSingleStep(0.1)
        self.spin_w_spread.setValue(5.0)
        widget_w_spread = QWidget()
        h_w_spread = QHBoxLayout(widget_w_spread)
        h_w_spread.setContentsMargins(0,0,0,0)
        h_w_spread.addWidget(self.spin_w_spread)
        self.lbl_score_spread = QLabel("0.00")
        h_w_spread.addWidget(self.lbl_score_spread)
        scoring_layout.addRow("Spread Weight:", widget_w_spread)

        # Kurtosis Weight
        self.spin_w_kurt = QDoubleSpinBox()
        self.spin_w_kurt.setRange(-1e2, 1e3)
        self.spin_w_kurt.setSingleStep(0.1)
        self.spin_w_kurt.setValue(0.1)
        widget_w_kurt = QWidget()
        h_w_kurt = QHBoxLayout(widget_w_kurt)
        h_w_kurt.setContentsMargins(0,0,0,0)
        h_w_kurt.addWidget(self.spin_w_kurt)
        self.lbl_score_kurt = QLabel("0.00")
        h_w_kurt.addWidget(self.lbl_score_kurt)
        scoring_layout.addRow("Kurtosis Weight:", widget_w_kurt)

        # Multi-peak Weight
        self.spin_w_multipeak = QDoubleSpinBox()
        self.spin_w_multipeak.setRange(-1e2, 1e3)
        self.spin_w_multipeak.setSingleStep(0.1)
        self.spin_w_multipeak.setValue(30.0)
        widget_w_multipeak = QWidget()
        h_w_multipeak = QHBoxLayout(widget_w_multipeak)
        h_w_multipeak.setContentsMargins(0,0,0,0)
        h_w_multipeak.addWidget(self.spin_w_multipeak)
        self.lbl_score_multipeak = QLabel("0.00")
        h_w_multipeak.addWidget(self.lbl_score_multipeak)
        scoring_layout.addRow("Multi-peak Weight:", widget_w_multipeak)

        # Low-E Penalty Weight
        self.spin_w_lowE = QDoubleSpinBox()
        self.spin_w_lowE.setRange(-1e2, 1e3)
        self.spin_w_lowE.setSingleStep(0.1)
        self.spin_w_lowE.setValue(-1.5)
        widget_w_lowE = QWidget()
        h_w_lowE = QHBoxLayout(widget_w_lowE)
        h_w_lowE.setContentsMargins(0,0,0,0)
        h_w_lowE.addWidget(self.spin_w_lowE)
        self.lbl_score_lowE = QLabel("0.00")
        h_w_lowE.addWidget(self.lbl_score_lowE)
        scoring_layout.addRow("Low-E Penalty Weight:", widget_w_lowE)

        # Overshoot Bonus Weight
        self.spin_w_overshoot = QDoubleSpinBox()
        self.spin_w_overshoot.setRange(-1e2, 1e3)
        self.spin_w_overshoot.setSingleStep(0.1)
        self.spin_w_overshoot.setValue(1.5)
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

        # --- Add Analyze Existing Images Button ---
        self.btn_analyze_existing = QPushButton('Analyze Existing Images')
        control_layout.addWidget(self.btn_analyze_existing)
        self.btn_analyze_existing.clicked.connect(self.start_batch_analysis)

        splitter.addWidget(control)
        self.plot_canvas = PlotCanvas(self)
        splitter.addWidget(self.plot_canvas)
        splitter.setSizes([350, 850])

        # Initial dummy calibration
        self.calib = {
            'maxWid':  1000,
            'maxLen':  1000,
            'projmat': np.eye(3),
            'pixel2mm': 0.1,
            'screen_to_axis_distance': 100.0,
            'dE_ds_interp': lambda x: np.interp(x, [0, 100, 200], [0, 1, 0]),
            'average_distance_covered': 1.0,
            'energy_interp': lambda x: np.interp(x, [0, 100, 200], [0, 100, 200]),
            'energy_partition': [0, 100, 200]
        }

        # Timer for batch processing
        self.batch_timer = None
        self.batch_files = []
        self.batch_index = 0
        self._batch_workers = []  # Track running batch workers

        self._live_workers = []  # Track running live workers
    
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

    def restart_observer(self):
        # Stop existing observer
        if hasattr(self, 'observer') and self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=1.0)  # Don't wait forever
            except Exception as ex:
                print(f"Warning: Error stopping previous observer: {ex}")
            finally:
                self.observer = None
        # Start new observer if all conditions are met
        if self.input_folder and self.output_folder and self.calib:
            try:
                handler = NewFileHandler(self)
                # Use PollingObserver for network folder compatibility
                self.observer = PollingObserver()
                self.observer.schedule(handler, self.input_folder, recursive=False)
                self.observer.start()
                try:
                    self.log.append(f'Watching {self.input_folder}')
                except RuntimeError:
                    # GUI destroyed while starting observer
                    if self.observer:
                        self.observer.stop()
                        self.observer = None
            except Exception as ex:
                try:
                    self.log.append(f"[ERROR] Failed to start observer: {ex}")
                except RuntimeError:
                    print(f"[ERROR] Failed to start observer: {ex}")

    def select_input_folder(self):
        print('[DEBUG] select_input_folder: Button clicked.')
        self.log.append('[DEBUG] select_input_folder: Button clicked.')
        try:
            path = QFileDialog.getExistingDirectory(self, 'Select Input Folder', options=QFileDialog.Option.ShowDirsOnly)
            print(f'[DEBUG] QFileDialog.getExistingDirectory returned: {path!r}')
            self.log.append(f'[DEBUG] QFileDialog.getExistingDirectory returned: {path!r}')
            if path:
                self.input_folder = path
                self.lbl_in.setText(self._elide_path(path, self.lbl_in))
                self.lbl_in.setToolTip(path)  # Show full path on hover
                print(f'[INFO] Input folder set: {path}')
                self.log.append(f'[INFO] Input folder set: {path}')
                self.restart_observer()
            else:
                print('[WARN] No folder selected or dialog was cancelled.')
                self.log.append('[WARN] No folder selected or dialog was cancelled.')
        except Exception as ex:
            print(f'[ERROR] Exception in select_input_folder: {ex}')
            self.log.append(f'[ERROR] Exception in select_input_folder: {ex}')

    def select_output_folder(self):
        print('[DEBUG] select_output_folder: Button clicked.')
        self.log.append('[DEBUG] select_output_folder: Button clicked.')
        try:
            path = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
            print(f'[DEBUG] QFileDialog.getExistingDirectory returned: {path!r}')
            self.log.append(f'[DEBUG] QFileDialog.getExistingDirectory returned: {path!r}')
            if path:
                self.output_folder = path
                self.lbl_out.setText(self._elide_path(path, self.lbl_out))
                self.lbl_out.setToolTip(path)  # Show full path on hover
                print(f'[INFO] Output folder set: {path}')
                self.log.append(f'[INFO] Output folder set: {path}')
                self.restart_observer()
            else:
                print('[WARN] No folder selected or dialog was cancelled.')
                self.log.append('[WARN] No folder selected or dialog was cancelled.')
        except Exception as ex:
            print(f'[ERROR] Exception in select_output_folder: {ex}')
            self.log.append(f'[ERROR] Exception in select_output_folder: {ex}')

    def _elide_path(self, path, label):
        """
        Truncate the path with ellipsis if it exceeds the label's max width.
        """
        metrics = label.fontMetrics()
        max_width = label.maximumWidth() if label.maximumWidth() > 0 else 180
        return metrics.elidedText(path, Qt.TextElideMode.ElideMiddle, max_width)

    def closeEvent(self, event):
        # Stop the file observer first to prevent new file processing
        if hasattr(self, 'observer') and self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2.0)  # Wait max 2 seconds
            except Exception as ex:
                print(f"Warning: Error stopping observer: {ex}")
            finally:
                self.observer = None
        # Stop all running batch workers
        for worker in getattr(self, '_batch_workers', []):
            if worker.isRunning():
                worker.quit()
                worker.wait(2000)  # Wait max 2 seconds per worker
        # Stop all running live workers
        for worker in getattr(self, '_live_workers', []):
            if worker.isRunning():
                worker.quit()
                worker.wait(2000)  # Wait max 2 seconds per worker
        # Set a flag to indicate the window is closing
        self._closing = True
        # Allow the event to proceed
        super().closeEvent(event)

    def start_batch_analysis(self):
        if not self.input_folder or not self.calib:
            self.log.append('[ERROR] Input folder and calibration must be set.')
            return
        # Find image files in input folder
        exts = ('.tif', '.tiff', '.png', '.jpg')
        files = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder)
                 if f.lower().endswith(exts)]
        if not files:
            self.log.append('[INFO] No image files found in input folder.')
            return
        self.log.append(f'[INFO] Found {len(files)} image files. Starting batch analysis at 1 Hz...')
        self.batch_files = files
        self.batch_index = 0
        if self.batch_timer:
            self.batch_timer.stop()
        self.batch_timer = QTimer(self)
        self.batch_timer.timeout.connect(self.process_next_batch_image)
        self.batch_timer.start(1000)  # 1 Hz

    def process_next_batch_image(self):
        if self.batch_index >= len(self.batch_files):
            self.log.append('[INFO] Batch analysis complete.')
            self.batch_timer.stop()
            return
        path = self.batch_files[self.batch_index]
        self.log.append(f'[INFO] Batch analyzing: {os.path.basename(path)}')
        worker = AnalysisWorker(self, path)
        worker.result_ready.connect(self.handle_batch_result)
        worker.error.connect(lambda msg: self.log.append(msg))
        worker.finished.connect(lambda: self._batch_workers.remove(worker) if worker in self._batch_workers else None)
        self._batch_workers.append(worker)
        worker.start()
        self.batch_index += 1

    def handle_batch_result(self, result):
        # Update feedback labels for scoring breakdown
        score_dict = result.get('score_dict', {})
        self.lbl_score_shape.setText(f"{score_dict.get('shape', 0.0):.2f}")
        self.lbl_score_peak.setText(f"{score_dict.get('peak', 0.0):.2f}")
        self.lbl_score_spread.setText(f"{score_dict.get('spread', 0.0):.2f}")
        self.lbl_score_kurt.setText(f"{score_dict.get('kurtosis', 0.0):.2f}")
        self.lbl_score_multipeak.setText(f"{score_dict.get('multipeak', 0.0):.2f}")
        self.lbl_score_lowE.setText(f"{score_dict.get('low', 0.0):.2f}")
        self.lbl_score_overshoot.setText(f"{score_dict.get('over', 0.0):.2f}")
        self.lbl_score_totalcost.setText(f"{score_dict.get('raw', 0.0):.2f}")
        # Log the cost and score
        self.log.append(f"[INFO] Batch result: Score {score_dict.get('score', 0.0):.2f} (raw cost={score_dict.get('raw', 0.0):.2f})")
        # Update plots
        self.plot_canvas.update_plots(
            post_PT=result.get('postPT'),
            spec2D=result.get('spec2D'),
            energy_axis=result.get('e_axis'),
            masked_energy=result.get('m_e'),
            masked_intensity=result.get('m_spec'),
            score_dict=score_dict,
            pixel2mm=self.calib['pixel2mm'],
        )

    def validate_parameters(self):
        if self.spin_minE.value() >= self.spin_maxE.value():
            self.log.append("[WARN] Min E must be less than Max E.")
            return False
        # Add more checks as needed
        return True

    def handle_new_file(self, path):
        """
        Called when a new image file is detected in the input folder.
        Starts analysis in a worker thread.
        """
        self.log.append(f"[INFO] Processing new file: {os.path.basename(path)}")
        worker = AnalysisWorker(self, path)
        worker.result_ready.connect(lambda result: self.handle_analysis_result(result, path))
        worker.error.connect(lambda msg: self.log.append(msg))
        worker.finished.connect(lambda: self._live_workers.remove(worker) if worker in self._live_workers else None)
        self._live_workers.append(worker)
        worker.start()

    def save_analysis_output(self, result, source_path):
        """
        Save analysis results to the output folder as CSV.
        result: dict containing keys 'score_dict', 'm_e', 'm_spec'
        source_path: original image file path (for naming)
        """
        import os
        score_dict = result.get('score_dict', {})
        m_e = result.get('m_e', [])
        m_spec = result.get('m_spec', [])
        # Calculate total counts in masked spectrum
        total_counts = np.sum(m_spec) if len(m_spec) > 0 else 0
        if self.output_folder:
            base = os.path.splitext(os.path.basename(source_path))[0]
            out_csv = os.path.join(self.output_folder, f"{base}_results.csv")
            try:
                with open(out_csv, "w") as f:
                    f.write("# Score Summary\n")
                    for k, v in score_dict.items():
                        f.write(f"{k},{v}\n")
                    # Add total counts line
                    f.write(f"# Total Counts in Masked Spectrum: {total_counts}\n")
                    f.write("# Masked Spectrum (Energy,Intensity)\n")
                    for e, s in zip(m_e, m_spec):
                        f.write(f"{e},{s}\n")
                self.log.append(f"[INFO] Results saved: {out_csv}")
            except Exception as ex:
                self.log.append(f"[ERROR] Failed to save results: {ex}")
        else:
            self.log.append("[WARN] Output folder not set. Results not saved.")

    def handle_analysis_result(self, result, source_path):
        # Update feedback labels for scoring breakdown
        score_dict = result.get('score_dict', {})
        self.lbl_score_shape.setText(f"{score_dict.get('shape', 0.0):.2f}")
        self.lbl_score_peak.setText(f"{score_dict.get('peak', 0.0):.2f}")
        self.lbl_score_spread.setText(f"{score_dict.get('spread', 0.0):.2f}")
        self.lbl_score_kurt.setText(f"{score_dict.get('kurtosis', 0.0):.2f}")
        self.lbl_score_multipeak.setText(f"{score_dict.get('multipeak', 0.0):.2f}")
        self.lbl_score_lowE.setText(f"{score_dict.get('low', 0.0):.2f}")
        self.lbl_score_overshoot.setText(f"{score_dict.get('over', 0.0):.2f}")
        self.lbl_score_totalcost.setText(f"{score_dict.get('raw', 0.0):.2f}")
        self.log.append(f"[INFO] Analysis result: Score {score_dict.get('score', 0.0):.2f} (raw cost={score_dict.get('raw', 0.0):.2f})")
        self.plot_canvas.update_plots(
            post_PT=result.get('postPT'),
            spec2D=result.get('spec2D'),
            energy_axis=result.get('e_axis'),
            masked_energy=result.get('m_e'),
            masked_intensity=result.get('m_spec'),
            score_dict=score_dict,
            pixel2mm=self.calib['pixel2mm'],
        )
        self.save_analysis_output(result, source_path)

    def process_next_batch_image(self):
        if self.batch_index >= len(self.batch_files):
            self.log.append('[INFO] Batch analysis complete.')
            self.batch_timer.stop()
            return
        path = self.batch_files[self.batch_index]
        self.log.append(f'[INFO] Batch analyzing: {os.path.basename(path)}')
        worker = AnalysisWorker(self, path)
        worker.result_ready.connect(lambda result: self.handle_analysis_result(result, path))
        worker.error.connect(lambda msg: self.log.append(msg))
        worker.finished.connect(lambda: self._batch_workers.remove(worker) if worker in self._batch_workers else None)
        self._batch_workers.append(worker)
        worker.start()
        self.batch_index += 1


if __name__ == '__main__':
    try:
        print('[DEBUG] Starting QApplication...')
        app = QApplication(sys.argv)
        print('[DEBUG] QApplication started.')
        w = MainWindow()
        print('[DEBUG] MainWindow created.')
        w.show()
        print('[DEBUG] MainWindow shown.')
        sys.exit(app.exec())
    except Exception as ex:
        import traceback
        print(f"[FATAL] Application crashed: {ex}")
        print(traceback.format_exc())
