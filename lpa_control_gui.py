# lpa_control_gui.py
# PyQt5-based control GUI for LPA Bayesian Optimization
# Includes threading, efficient plotting, file-watcher placeholders,
# error handling, persistent logging, mode switching, and full integration.

import sys
import xopt
print("Python executable:", sys.executable)
print("xopt location:", xopt.__file__)
print("xopt version:", xopt.__version__)

import sys              # System-specific parameters and functions
import os               # Miscellaneous operating system interfaces
import glob             # Unix style pathname pattern expansion
import time             # Time access and conversions
import csv              # CSV file reading and writing
from threading import Thread, Event

from bo_engine import create_xopt  # Custom Bayesian Optimization engine

from PyQt5.QtWidgets import (
    QApplication,       # Core application object
    QMainWindow,        # Main window class
    QWidget,            # Basic UI container
    QVBoxLayout,        # Vertical and horizontal layout managers
    QHBoxLayout,
    QPushButton,        # Clickable button widget
    QLabel,             # Display text
    QLineEdit,          # Single-line text entry
    QSpinBox,           # Numeric spin box
    QFileDialog,        # File selection dialog
    QMessageBox,        # Popup messages and dialogs
    QCheckBox,          # Checkbox widget
    QComboBox,          # Dropdown selection widget
    QSlider,            # Slider widget for manual control
    QGridLayout,        # Grid layout manager for complex arrangements
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSettings, QThread # Added QSettings
import pyqtgraph as pg  # High-performance plotting library for PyQt
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QFontDatabase  # Font database for custom fonts

# Configure pyqtgraph aesthetics
pg.setConfigOption('background', '#f0f8ff')  # Set plot background to AliceBlue
pg.setConfigOption('foreground', '#006064')  # Set plot axes/text to Dark Cyan

class WorkerSignals(QObject):
    """
    Defines custom signals for worker threads:
    - new_metrics: emits a dict of updated metrics (e.g., loss, accuracy)
    - new_suggestions: emits a dict of new parameter suggestions from BO
    - finished: signals that the worker thread has completed its task
    """
    new_metrics = pyqtSignal(dict)           
    new_suggestions = pyqtSignal(dict)
    finished = pyqtSignal()
    # param_confirmed = pyqtSignal(dict)  # Signal for parameter confirmation
    param_request   = pyqtSignal(dict)   # request GUI to confirm
    param_confirmed = pyqtSignal(dict)

class OptimizationWorker(QThread):
    """
    Runs in the background to:
     1. Propose new control-knob settings using Xopt
     2. Wait for you to apply them and confirm
     3. Read back the shot results (metrics)
     4. Send those results for plotting/logging
     5. Update suggestions using Bayesian Optimization
    """
    def __init__(self, gui, stop_event):
        super().__init__()
        self.gui = gui
        self.signals = WorkerSignals()
        self.stop_event = stop_event
        self.param_event = Event()  # Event for parameter confirmation
        self.confirmed_params = None

        # Prepare a CSV log file in the working folder
        self.log_file = os.path.join(os.getcwd(), 'params_metrics_log.csv')
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['shot','params','spec','charge','stability','timestamp'])

        # Initialize Xopt optimizer with active knobs and evaluation function
        mode   = self.gui.get_acquisition_mode()
        bounds = self.gui.get_bounds()
        self.X  = create_xopt(
            self.gui.get_active_params(),
            self.evaluate,
            bounds=bounds,
            acquisition_mode=mode
        )

    def run(self):
        shot = 0

        # Ensure initial data is available for Xopt
        if getattr(self.X, "data", None) is None or len(self.X.data) == 0:
            self.X.random_evaluate()

        # Main loop: keep going until user stops
        while not self.stop_event.is_set():
            shot += 1

            # Generate new suggestions using Bayesian optimization
            step = self.X.step()
            if step is None or not hasattr(step, 'data') or step.data is None:
                print("[OptimizationWorker] Xopt.step() returned None or invalid step. Stopping optimization.")
                break
            suggestions = dict(step.data.iloc[-1])

            # Round suggestions to nearest integer + 1 decimal place
            for key, value in suggestions.items():
                if isinstance(value, (int, float)):
                    suggestions[key] = round(value, 1)
        
            # Tell GUI: here are the next settings to try
            self.signals.new_suggestions.emit(suggestions)

            # Wait for parameter confirmation via signal/event
            self.param_event.clear()
            # self.gui.request_param_confirmation(suggestions)
            self.signals.param_request.emit(suggestions)
            while not self.param_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)
            if self.stop_event.is_set():
                break
            params = self.confirmed_params.copy() if self.confirmed_params else suggestions.copy()

            # --- ROUND PARAMS BEFORE ADDING TO Xopt HISTORY ---
            for k, v in params.items():
                if isinstance(v, float):
                    params[k] = round(v, 1)

            # TODO: actually push these settings to your hardware

            # Read back the last few shots’ data (up to a timeout)
            try:
                metrics = self.collect_metrics(timeout=5)
            except Exception as e:
                metrics = {'overall': None, 'spec': None, 'charge': None}
            if metrics is None:
                metrics = {'overall': None, 'spec': None, 'charge': None}

            # Build multi-objective result for Xopt
            # (spectra_score, charge, stability are returned by collect_metrics)

            # Send the metrics to the GUI for real-time plotting
            self.signals.new_metrics.emit(metrics)

            # Append this shot’s info to the CSV log
            try:
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        shot,
                        str(params),
                        metrics.get('spec'),
                        metrics.get('charge'),
                        metrics.get('stability'),
                        time.time()
                    ])
            except Exception as e:
                print(f"Error writing to log file: {e}")

            # Store the evaluated result into the Xopt history (three objectives)
            self.X.add_data({**params,
                             'spectra_score': metrics.get('spec'),
                             'charge':       metrics.get('charge'),
                             'stability':    metrics.get('stability')})

        # Signal that we’ve fully stopped
        self.signals.finished.emit()

    def evaluate(self, inputs_list):
        """
        Receives suggestions from Xopt, pushes them through GUI + measurement,
        and returns a list of dicts with multi-objective metrics.
        """
        results = []
        print(f"[evaluate] Received inputs_list of type {type(inputs_list)}")
        # If inputs_list is a dict, convert to list of dicts
        if isinstance(inputs_list, dict):
            print(f"[evaluate] inputs_list is a dict with keys: {list(inputs_list.keys())}")
            # Try to interpret as a single parameter set
            # If all values are scalars, treat as one param set
            if all(isinstance(v, (int, float, str)) for v in inputs_list.values()):
                inputs_list = [inputs_list]
                print(f"[evaluate] Converted dict to single-item list: {inputs_list}")
            else:
                # If values are lists, try to build param dicts for each
                try:
                    param_names = list(inputs_list.keys())
                    param_values = list(inputs_list.values())
                    n_items = len(param_values[0])
                    inputs_list = [dict(zip(param_names, [pv[i] for pv in param_values])) for i in range(n_items)]
                    print(f"[evaluate] Converted dict of lists to list of dicts: {inputs_list}")
                except Exception as e:
                    print(f"[evaluate] Could not convert dict to list of dicts: {e}")
                    inputs_list = []

        print(f"[evaluate] Final inputs_list type: {type(inputs_list)}, length: {len(inputs_list) if hasattr(inputs_list, 'len') else 'N/A'}")
        for idx, params in enumerate(inputs_list):
            print(f"[evaluate] Processing input {idx}: type={type(params)}, value={params}")

            if not isinstance(params, dict):
                import ast # Safely convert string to dict
                try:
                    params = ast.literal_eval(params)
                    print(f"[evaluate] Converted parameters to dict: {params}")
                except Exception as e:
                    print(f"[evaluate] Failed to convert parameters: {params}, error: {e}")
                    continue  # skip if cannot convert
            
            # --- ROUND PARAMS TO 1 DECIMAL PLACE ---
            params = {k: round(v, 1) if isinstance(v, float) else v for k, v in params.items()}

            self.param_event.clear()
            self.signals.param_request.emit(params)

            print(f"[evaluate] Waiting for user to confirm parameters: {params}")
            while not self.param_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)
            if self.stop_event.is_set():
                print("[evaluate] Stop event set, breaking loop.")
                break
            try:
                metrics = self.collect_metrics(timeout=5)
            except Exception as e:
                print(f"[evaluate] Exception in collect_metrics: {e}")
                metrics = {'spec': None, 'charge': None, 'stability': None}

            print(f"[evaluate] metrics after collect_metrics: type={type(metrics)}, value={metrics}")
            # Convert pandas Series or DataFrame to dict if needed
            if metrics is None:
                print(f"[evaluate] Metrics is None, skipping.")
                continue
            if isinstance(metrics, (list, tuple)):
                print(f"[evaluate] Metrics is a list/tuple, skipping: {metrics}")
                continue
            if hasattr(metrics, "to_dict") and not isinstance(metrics, dict):
                print(f"[evaluate] Metrics has to_dict, converting.")
                metrics = metrics.to_dict()
            if not isinstance(metrics, dict):
                print(f"[evaluate] Warning: metrics is not a dict after conversion, skipping. Type: {type(metrics)} Value: {metrics}")
                continue
            # Final check for required keys
            for key in ('spec', 'charge', 'stability'):
                if key not in metrics:
                    print(f"[evaluate] Warning: metrics missing key '{key}', value: {metrics}")
            result_dict = {
                'spectra_score': metrics.get('spec'),
                'charge':       metrics.get('charge'),
                'stability':    metrics.get('stability')
            }
            print(f"[evaluate] Appending result dict: {result_dict}")
            results.append(result_dict)
        print(f"[evaluate] Returning results list of length {len(results)}: {results}")
        return results

    def on_param_confirmed(self, params):
        self.confirmed_params = params
        self.param_event.set()

    def collect_metrics(self, timeout=5):
        """
        Look in the measurement folders for your latest E-Spec or Profile & ICT data.
        - In E-Spec mode: read score and raw from summary lines in .csv files, or score,cost,counts from .txt files.
        - In Profile & ICT mode: fallback to spec from profile_dir and charge from ict_dir.
        Returns a dict when both spec & charge are ready, or None after timeout.
        Computes mean and std over last N shots for the current parameter set.
        """
        import numpy as np
        import time
        start = time.time()
        while time.time() - start < timeout:

            mode = self.gui.get_mode()  # "E-Spec" or "Profile & ICT"
            overall = None  # will be set below

            spec = None
            charge = None
            spec_std = None
            charge_std = None

            N = self.gui.shots_spin.value()
            print(f"Collecting metrics for mode '{mode}' with N={N}...")

            if mode == 'E-Spec':
                print("Using E-Spec mode")
                path = self.gui.espec_dir
                # Gather latest N output files (.txt and .csv)
                txt_files = sorted(glob.glob(os.path.join(path, '*.txt')))
                csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))
                # Use only the latest N files (prefer txt, then csv)
                all_files = txt_files + csv_files
                all_files = sorted(all_files)[-N:]
                if len(all_files)< N:
                    # Log only once per wait
                    if not hasattr(self, '_waiting_for_files') or not self._waiting_for_files:
                        self.gui.log_activity(f"Waiting for at least {N} files in {path} (found {len(all_files)}).")
                        self._waiting_for_files = True
                    time.sleep(0.2)
                    continue
                else:
                    self._waiting_for_files = False
                # Print notification if new files are detected
                if hasattr(self, '_last_seen_files'):
                    new_files = set(all_files) - set(self._last_seen_files)
                    if new_files:
                        for fn in new_files:
                            # Print last 3 subfolder addresses
                            parts = os.path.normpath(fn).split(os.sep)
                            subpath = os.sep.join(parts[-4:]) if len(parts) >= 4 else fn
                            print(f"[File Detected] New E-Spec file: .../{subpath}")
                self._last_seen_files = list(all_files)
                scores, charges = [], []
                for fn in all_files:
                    try:
                        if fn.endswith('.txt'):
                            line = open(fn, 'r').readline().strip()
                            sc, cost, ct = map(float, line.split(','))  # score,cost,counts
                            scores.append(sc)
                            charges.append(ct)
                        elif fn.endswith('.csv'):
                            score_val = None
                            charge_val = None
                            with open(fn, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line or line.startswith('#'):
                                        continue
                                    parts = line.split(',')
                                    if len(parts) != 2:
                                        continue
                                    key, value = parts[0].strip().lower(), parts[1].strip()
                                    if key == 'score' and score_val is None:
                                        try:
                                            score_val = float(value)
                                        except Exception:
                                            pass
                                    elif key == 'raw' and charge_val is None:
                                        try:
                                            charge_val = float(value)
                                        except Exception:
                                            pass
                                    if score_val is not None and charge_val is not None:
                                        break
                            if score_val is not None:
                                scores.append(score_val)
                            if charge_val is not None:
                                charges.append(charge_val)
                    except Exception as e:
                        print(f"Error parsing E-Spec file {fn}: {e}")
                if scores and charges:
                    spec = float(np.mean(scores))
                    charge = float(np.mean(charges))
                    spec_std = float(np.std(scores))
                    charge_std = float(np.std(charges))
            else:
                # ----- Profile & ICT branch (unchanged) -----
                # spec from profile_dir
                prof = self.gui.profile_dir
                prof_files = sorted(glob.glob(os.path.join(prof, '*.txt')))
                if len(prof_files) >= N:
                    try:
                        vals = [float(open(f).readline()) for f in prof_files[-N:]]
                        spec = float(np.mean(vals))
                        spec_std = float(np.std(vals))
                    except Exception as e:
                        print(f"Error reading profile spec files: {e}")

                # charge from ict_dir
                ict = self.gui.ict_dir
                ict_files = sorted(glob.glob(os.path.join(ict, '*.txt')))
                if len(ict_files) >= N:
                    try:
                        vals = [float(open(f).readline()) for f in ict_files[-N:]]
                        charge = float(np.mean(vals))
                        charge_std = float(np.std(vals))
                    except Exception as e:
                        print(f"Error reading ICT charge files: {e}")

            # Once we have both metrics, compute stability and return them
            if spec is not None and charge is not None:
                params = tuple(sorted(self.gui.confirmed_params.items())) if self.gui.confirmed_params else None
                if not hasattr(self, '_pareto_param_history_worker'):
                    self._pareto_param_history_worker = {}
                if params is not None:
                    entry = self._pareto_param_history_worker.setdefault(params, [])
                    entry.append((spec, charge))
                    if len(entry) > N:
                        entry[:] = entry[-N:]
                    arr = np.array(entry)
                    mean_spec = float(np.mean(arr[:,0]))
                    stability = float(np.std(arr[:,0]) / mean_spec) if mean_spec != 0 else 0.0
                else:
                    stability = 0.0
                # Compute geometric mean for 'overall' metric
                if spec > 0 and charge > 0:
                    overall = float(np.sqrt(spec * charge))
                else:
                    overall = 0.0
                # Return mean and std for plotting error bars
                result = {'overall': overall, 'spec': spec, 'charge': charge, 'spec_std': spec_std, 'charge_std': charge_std, 'stability': stability}
                print("Returning metrics dict:", result)  # Debug print
                # If result is accidentally a Series, convert to dict
                if hasattr(result, "to_dict") and not isinstance(result, dict):
                    print("Converting result to dict")
                    result = result.to_dict()
                return result
            
            time.sleep(0.1)

        # Timeout: no full data set available
        print("Returning None from collect_metrics")
        return None

class LPAControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Main window setup: title and size
        self.setWindowTitle("LPA Bayesian Optimization Control")
        self.resize(1400, 900)

        # ---------------------------------------------------------
        # State variables
        # ---------------------------------------------------------
        self.confirmed_params = None     # Holds the last user‐confirmed knob settings
        self.is_running = False          # True while optimization loop is active
        self.history = {                 # Stores shot‐by‐shot values for plotting
            'overall': [], 'spec': [], 'charge': []
        }
        self.shot_number = 0             # Counter for how many shots we've processed
        self.stop_event = Event()        # Signal to stop the background thread
        self.worker = None               # Reference to the optimization thread
        self.settings = QSettings('LPAControl', 'UserSettings')

        # ---------------------------------------------------------
        # Layout containers
        # ---------------------------------------------------------
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        self.setCentralWidget(w)

        # ---------------------------------------------------------
        # Main layout: horizontal split (left controls, parameter table, action buttons, activity log)
        # ---------------------------------------------------------
        main_hbox = QHBoxLayout()
        l.addLayout(main_hbox)

        # --- Left column: Directory buttons, diagnostics mode ---
        left_col = QVBoxLayout()
        left_col.setSpacing(15)
        for key, label in [('espec', 'E-Spec'), ('profile', 'Profile'), ('ict', 'ICT')]:
            btn = QPushButton(f"Set {label} Dir")
            btn.setFixedWidth(120)
            btn.clicked.connect(lambda _, k=key: self.choose_folder(k))
            left_col.addWidget(btn, alignment=Qt.AlignLeft)
        left_col.addSpacing(10)
        left_col.addWidget(QLabel("Diagnostics Mode:"), alignment=Qt.AlignLeft)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["E-Spec", "Profile & ICT"])
        left_col.addWidget(self.mode_combo, alignment=Qt.AlignLeft)
        left_col.addStretch(1)
        main_hbox.addLayout(left_col)

        # --- Vertical line after left column ---
        from PyQt5.QtWidgets import QFrame, QTextEdit
        vline1 = QFrame()
        vline1.setFrameShape(QFrame.VLine)
        vline1.setFrameShadow(QFrame.Sunken)
        main_hbox.addWidget(vline1)

        # --- Parameter table with checkboxes aligned ---
        param_names = ['plasma_z', 'gas_pressure', 'GDD_phi2', 'blade_x', 'blade_z']
        grid = QGridLayout()
        cols = ["", "Lower", "Upper", "Parameter", "Suggest", "Use"]
        for c, title in enumerate(cols):
            lbl = QLabel(title)
            if title:
                lbl.setStyleSheet("font-weight:bold")
            grid.addWidget(lbl, 0, c, alignment=Qt.AlignLeft)
        self.param_checks = {}
        self.param_widgets = {}
        self.param_bounds  = {}
        for r, name in enumerate(param_names, start=1):
            # Checkbox
            cb = QCheckBox()
            cb.setChecked(True)
            grid.addWidget(cb, r, 0, alignment=Qt.AlignLeft)
            self.param_checks[name] = cb
            # Lower bound box
            lo = QLineEdit("0.0")
            lo.setFixedWidth(70)
            grid.addWidget(lo, r, 1, alignment=Qt.AlignLeft)
            # Upper bound box
            hi = QLineEdit("1.0")
            hi.setFixedWidth(70)
            grid.addWidget(hi, r, 2, alignment=Qt.AlignLeft)
            # Parameter name
            grid.addWidget(QLabel(name), r, 3, alignment=Qt.AlignLeft)
            # Suggested value (read-only, 90px wide)
            sug = QLineEdit()
            sug.setReadOnly(True)
            sug.setFixedWidth(90)
            grid.addWidget(sug, r, 4, alignment=Qt.AlignLeft)
            # “Use” value (90px wide)
            val = QLineEdit()
            val.setFixedWidth(90)
            # Connect editingFinished to highlight field if edited after copy
            val.editingFinished.connect(lambda n=name, w=val: self._on_use_field_edited(n, w))
            grid.addWidget(val, r, 5, alignment=Qt.AlignLeft)
            self.param_bounds[name]  = (lo, hi)
            self.param_widgets[name] = (sug, val)
        # Add the parameter grid to the main layout
        main_hbox.addLayout(grid, stretch=2)

        # --- Vertical line before action buttons column ---
        vline2 = QFrame()
        vline2.setFrameShape(QFrame.VLine)
        vline2.setFrameShadow(QFrame.Sunken)
        main_hbox.addWidget(vline2)

        # --- Action buttons in a vertical column ---
        btn_col = QVBoxLayout()
        btn_col.setSpacing(12)
        self.copy_btn = QPushButton("Copy Suggestions")
        self.confirm_btn = QPushButton("Confirm Parameters")
        self.start_btn = QPushButton("Start Optimization")
        self.pause_btn = QPushButton("Pause")
        self.restart_btn = QPushButton("Restart")
        self.show_params_btn = QPushButton("Show Current Parameters")
        btns = [self.copy_btn, self.confirm_btn, self.start_btn, self.pause_btn, self.restart_btn, self.show_params_btn]
        max_width = max(btn.sizeHint().width() for btn in btns)
        for btn in btns:
            btn.setFixedWidth(max_width)
        self.copy_btn.clicked.connect(self.copy_suggestions_to_use)
        btn_col.addWidget(self.copy_btn, alignment=Qt.AlignLeft)
        self.confirm_btn.clicked.connect(self.confirm_parameters)
        self.confirm_btn.setEnabled(False)
        btn_col.addWidget(self.confirm_btn, alignment=Qt.AlignLeft)
        self.start_btn.clicked.connect(self.start_optimization)
        self.start_btn.setEnabled(False)   # disabled until dirs are set
        btn_col.addWidget(self.start_btn, alignment=Qt.AlignLeft)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_optimization)
        btn_col.addWidget(self.pause_btn, alignment=Qt.AlignLeft)
        self.restart_btn.clicked.connect(self.restart_optimization)
        btn_col.addWidget(self.restart_btn, alignment=Qt.AlignLeft)
        self.show_params_btn.clicked.connect(self.show_current_parameters)
        btn_col.addWidget(self.show_params_btn, alignment=Qt.AlignLeft)
        btn_col.addStretch(1)
        main_hbox.addLayout(btn_col)

        # --- Activity log immediately to the right of action buttons ---
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Activity Log...")
        self.activity_log.setMinimumWidth(220)
        main_hbox.addWidget(self.activity_log, stretch=1)

        # --- Add stretch to push everything left and open space on right (optional) ---
        # main_hbox.addStretch(1)  # You can comment this out if not needed

        # --- Below: Manual Explore/Exploit slider, shots spin, plots ---
        l.addSpacing(10)
        manual_layout = QHBoxLayout()
        self.manual_checkbox = QCheckBox("Manual mode")
        manual_layout.addWidget(self.manual_checkbox)
        self.explore_label_left  = QLabel("Explore")
        self.explore_label_left.setVisible(False)
        manual_layout.addWidget(self.explore_label_left)
        self.explore_slider = QSlider(Qt.Horizontal)
        self.explore_slider.setRange(0, 2)
        self.explore_slider.setValue(1)
        self.explore_slider.setTickPosition(QSlider.TicksBelow)
        self.explore_slider.setTickInterval(1)
        self.explore_slider.setEnabled(False)
        self.explore_slider.setVisible(False)
        self.explore_slider.setFixedWidth(120)
        manual_layout.addWidget(self.explore_slider)
        self.explore_label_right = QLabel("Exploit")
        self.explore_label_right.setVisible(False)
        manual_layout.addWidget(self.explore_label_right)
        manual_layout.addStretch(1)
        l.addLayout(manual_layout)
        self.manual_checkbox.toggled.connect(self._on_manual_toggled)

        # Move 'Shots to average' to the lowest row of the first column
        shots_hbox = QHBoxLayout()
        shots_hbox.addWidget(QLabel("Shots to average:"), alignment=Qt.AlignLeft)
        self.shots_spin = QSpinBox()
        self.shots_spin.setMinimum(1)
        self.shots_spin.setValue(10)
        shots_hbox.addWidget(self.shots_spin, alignment=Qt.AlignLeft)
        shots_hbox.addStretch(1)
        left_col.addSpacing(10)
        left_col.addLayout(shots_hbox)

        # --- Plots: overall, spec, charge in a vertical column, Pareto plot to the right ---
        plot_hbox = QHBoxLayout()
        plot_vbox = QVBoxLayout()
        self.plot_overall = pg.PlotWidget(title="Overall Objective")
        self.plot_spec    = pg.PlotWidget(title="Diagnostic Score")
        self.plot_charge  = pg.PlotWidget(title="Charge Metric")
        for p in [self.plot_overall, self.plot_spec, self.plot_charge]:
            p.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
            plot_vbox.addWidget(p)
        plot_vbox.addStretch(1)
        plot_hbox.addLayout(plot_vbox, stretch=2)
        # Pareto plot immediately to the right of the three main plots
        from pyqtgraph import PlotWidget, ColorBarItem
        self.pareto_plot = PlotWidget(title="Pareto Front: Spec vs Charge (color=Stability)")
        self.pareto_plot.setLabel('bottom', 'Spectra Score')
        self.pareto_plot.setLabel('left', 'Charge')
        self.pareto_scatter = self.pareto_plot.plot([], [], pen=None, symbol='o', symbolBrush=None)
        self.pareto_colorbar = None  # Placeholder for colorbar
        plot_hbox.addWidget(self.pareto_plot, stretch=1)
        l.addLayout(plot_hbox)

        # --- Interactivity: click on Pareto plot to show parameter settings ---
        self.pareto_plot.scene().sigMouseClicked.connect(self._on_pareto_click)

        # --- Reset Plots/Analysis button ---
        self.reset_btn = QPushButton("Reset Plots/Analysis")
        self.reset_btn.clicked.connect(self.reset_plots_and_analysis)
        l.addWidget(self.reset_btn, alignment=Qt.AlignRight)

        pen = pg.mkPen('#00796b', width=2)
        brush = '#009688'
        self.curves = {
            'overall': self.plot_overall.plot(pen=pen, symbol='o', symbolBrush=brush),
            'spec':    self.plot_spec.plot(pen=pen, symbol='o', symbolBrush=brush),
            'charge':  self.plot_charge.plot(pen=pen, symbol='o', symbolBrush=brush),
        }

        # Data for Pareto plot
        self.pareto_data = []  # List of dicts: {'spec':..., 'charge':..., 'stability':..., 'params':...}

        # Restore settings from QSettings
        self.restore_settings()

    def resizeEvent(self, ev):

        """Keep slider at half the window width on resize."""
        super().resizeEvent(ev)
        # only adjust if the slider exists
        if hasattr(self, 'explore_slider'):
            self.explore_slider.setFixedWidth(self.width() // 2)
            
    def log_activity(self, message):
        """Append a timestamped message to the activity log."""
        from datetime import datetime
        ts = datetime.now().strftime('%H:%M:%S')
        self.activity_log.append(f"[{ts}] {message}")

    def copy_suggestions_to_use(self):
        """Copy the suggested values into the 'Use' input boxes and reset highlight, only for checked parameters."""
        for name, (sug, use) in self.param_widgets.items():
            if self.param_checks[name].isChecked():
                use.setText(sug.text())
                use.setStyleSheet("background: white;")  # Reset background
        self._use_fields_edited = set()  # Track which fields have been edited
        self.log_activity("Copied suggested parameters to 'Use' boxes for active parameters.")

    def _on_use_field_edited(self, name, widget):
        """Highlight the 'Use' field if edited after copying suggestions."""
        widget.setStyleSheet("background: #fff59d;")  # Light yellow
        if not hasattr(self, '_use_fields_edited'):
            self._use_fields_edited = set()
        self._use_fields_edited.add(name)

    def _confirm_param(self, name, widget):
        """Store the float you typed when you press Enter in the 'Use:' box.""" 
        try:
            val = float(widget.text())
            if self.confirmed_params is None:
                self.confirmed_params = {}
            self.confirmed_params[name] = val
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                f"Please enter a number for {name}.")

    def choose_folder(self, key):
        """Open a folder dialog and save the selected path to self.<key>_dir."""
        path = QFileDialog.getExistingDirectory(
            self, f"Select {key} directory"
        )
        if path:
            self._update_start_button_state()
            setattr(self, f"{key}_dir", path)
            self.log_activity(f"Set {key} directory: {path}")
            QMessageBox.information(self, "Path Set", f"{key} dir: {path}")

    def _update_start_button_state(self):
            """Enable Start only if the needed dirs are set."""
            mode = self.get_mode()
            if mode == 'E-Spec':
                ready = hasattr(self, 'espec_dir')
            else:
                ready = hasattr(self, 'profile_dir') and hasattr(self, 'ict_dir')
            self.start_btn.setEnabled(ready)

    def start_optimization(self):
        """Kick off the background OptimizationWorker thread."""
        self.stop_event.clear()
        worker = OptimizationWorker(self, self.stop_event)
        worker.signals.new_metrics.connect(self.record_and_plot)
        worker.signals.new_suggestions.connect(self.update_suggestions)
        worker.signals.param_request.connect(self.request_param_confirmation)
        worker.start()
        self.signals = worker.signals  # keep for confirm
        self.worker = worker
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.restart_btn.setEnabled(False)
        self.log_activity("Optimization started.")

    def pause_optimization(self):
        """Temporarily pause sending new suggestions to hardware."""
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.log_activity("Optimization paused.")

    def restart_optimization(self):
        """
        Fully stop the worker, clear past data & plots,
        and re-enable Start.
        """
        self.stop_event.set()
        if self.worker:
            self.worker.join()
        self.is_running = False
        # Clear history and plot curves
        for hist in self.history.values():
            hist.clear()
        for c in self.curves.values():
            c.setData([], [])
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.log_activity("Optimization restarted and history cleared.")

    def get_active_params(self):
        """Return list of parameter names you’ve checked as active."""
        return [n for n, cb in self.param_checks.items() if cb.isChecked()] 

    def _on_manual_toggled(self, checked: bool):
        """Show or hide the explore/exploit slider when manual mode is toggled."""
        self.explore_slider.setEnabled(checked)
        self.explore_slider.setVisible(checked)
        self.explore_label_left.setVisible(checked)
        self.explore_label_right.setVisible(checked)

    def get_bounds(self):
        """
        Read the lower/upper QLineEdits for each active parameter.
        Returns a dict {name: [min, max]}.
        """
        bounds = {}
        for name, (lo_edit, hi_edit) in self.param_bounds.items():
            if self.param_checks[name].isChecked():
                try:
                    lo = float(lo_edit.text())
                    hi = float(hi_edit.text())
                    if hi <= lo:
                        raise ValueError("upper must be > lower")
                    bounds[name] = [lo, hi]
                except ValueError as e:
                    QMessageBox.warning(self, "Invalid Bounds",
                                        f"Bounds for {name} invalid: {e}\n"
                                        "Using [0.0, 1.0].")
                    bounds[name] = [0.0, 1.0]
        return bounds

    def get_acquisition_mode(self):
        """
        Return 'explore'|'balanced'|'exploit' based on the 3-point slider,
        or 'balanced' if manual mode is off.
        """
        if hasattr(self, 'manual_checkbox') and self.manual_checkbox.isChecked():
            v = self.explore_slider.value()
            if v == 0:
                return 'explore'
            elif v == 2:
                return 'exploit'
            else:
                return 'balanced'
        return 'balanced'

    def get_mode(self):
        """Return the current diagnostics mode from the dropdown."""
        mode = self.mode_combo.currentText()
        if not hasattr(self, '_last_mode'):
            self._last_mode = mode
        if mode != self._last_mode:
            self.log_activity(f"Diagnostics mode changed: {self._last_mode} → {mode}")
            self._last_mode = mode
        return mode

    def update_suggestions(self, sugg):
        """
        When the worker emits new_suggestions:
         - Clear any previously confirmed values
         - Display the new suggested values in the left-hand boxes
        """
        self.confirmed_params = None
        for name, val in sugg.items():
            # Only round numeric values
            if isinstance(val, (int, float)):
                val = round(val, 1)
            if name in self.param_widgets:
                self.param_widgets[name][0].setText(str(val))
        self.confirm_btn.setEnabled(True)

    def record_and_plot(self, metrics):
        """
        When the worker emits new_metrics:
         - Increment shot counter
         - Append each metric to history
         - Re-draw the curves with the updated data
         - Update Pareto data and plot
        """
        self.shot_number += 1
        for key in ['overall', 'spec', 'charge']:
            val = metrics.get(key)
            if val is not None:
                self.history[key].append((self.shot_number, val))
                if self.history[key]:
                    xs, ys = zip(*self.history[key])
                    self.curves[key].setData(xs, ys)
                else:
                    self.curves[key].setData([], [])

        # --- Pareto data pipeline ---
        # Track repeated measurements for each parameter set
        if not hasattr(self, '_pareto_param_history'):
            self._pareto_param_history = {}  # key: param tuple, value: list of (spec, charge)
        # Get current parameter set (as a tuple for dict key)
        params = tuple(sorted(self.confirmed_params.items())) if self.confirmed_params else None
        if params is not None:
            # Add this measurement to the history for this parameter set
            entry = self._pareto_param_history.setdefault(params, [])
            entry.append((metrics.get('spec'), metrics.get('charge')))
            # Only keep the last N (shots_spin) measurements for stability calculation
            N = self.shots_spin.value()
            if len(entry) > N:
                entry[:] = entry[-N:]
            # Compute mean spec, mean charge, and stability (coefficient of variation of spec)
            import numpy as np
            arr = np.array(entry)
            mean_spec = float(np.mean(arr[:,0]))
            mean_charge = float(np.mean(arr[:,1]))
            # Stability: coefficient of variation (std/mean) of spec (or charge)
            stability = float(np.std(arr[:,0]) / mean_spec) if mean_spec != 0 else 0.0
            # Update or add to pareto_data (replace if param set already exists)
            found = False
            for d in self.pareto_data:
                if d['params'] == params:
                    d['spec'] = mean_spec
                    d['charge'] = mean_charge
                    d['stability'] = stability
                    found = True
                    break
            if not found:
                self.pareto_data.append({'spec': mean_spec, 'charge': mean_charge, 'stability': stability, 'params': params})
            self.update_pareto_plot()

    def confirm_parameters(self):
        params = {}
        for name, (sug, use) in self.param_widgets.items():
            text = use.text().strip()
            if text == "":
                QMessageBox.warning(self, "Missing Value",
                                    f"Please enter a value for '{name}' before confirming.")
                return
            try:
                val = float(text)
            except ValueError:
                QMessageBox.warning(self, "Invalid Number",
                                    f"Could not parse '{text}' as a number for '{name}'.")
                return
            # Enforce bounds
            lo = float(self.param_bounds[name][0].text())
            hi = float(self.param_bounds[name][1].text())
            if not (lo < val < hi):
                QMessageBox.warning(self, "Out of Bounds",
                                    f"Value for '{name}' must be between {lo} and {hi}.")
                return
            params[name] = val
        self.confirmed_params = params
        self.confirm_btn.setEnabled(False)
        self.log_activity(f"Parameters confirmed: {params}")
        # Notify worker via signal/event
        if hasattr(self, '_pending_worker') and self._pending_worker:
            self._pending_worker.on_param_confirmed(params)
            self._pending_worker = None

    def restore_settings(self):
        # Restore directories
        for key in ['espec', 'profile', 'ict']:
            val = self.settings.value(f'{key}_dir', None)
            if val:
                setattr(self, f'{key}_dir', val)
        # Restore parameter bounds
        for name in ['plasma_z', 'gas_pressure', 'GDD_phi2', 'blade_x', 'blade_z']:
            lo = self.settings.value(f'{name}_lo', None)
            hi = self.settings.value(f'{name}_hi', None)
            if lo is not None and hi is not None:
                self.param_bounds[name][0].setText(str(lo))
                self.param_bounds[name][1].setText(str(hi))

    def save_settings(self):
        for key in ['espec', 'profile', 'ict']:
            val = getattr(self, f'{key}_dir', None)
            if val:
                self.settings.setValue(f'{key}_dir', val)
        for name in ['plasma_z', 'gas_pressure', 'GDD_phi2', 'blade_x', 'blade_z']:
            lo, hi = self.param_bounds[name]
            self.settings.setValue(f'{name}_lo', lo.text())
            self.settings.setValue(f'{name}_hi', hi.text())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def request_param_confirmation(self, suggestions):
        # Called by worker to request user confirmation
        self.update_suggestions(suggestions)
        self.confirm_btn.setEnabled(True)
        self._pending_worker = self.worker

    def confirm_parameters(self):
        params = {}
        for name, (sug, use) in self.param_widgets.items():
            text = use.text().strip()
            if text == "":
                QMessageBox.warning(self, "Missing Value",
                                    f"Please enter a value for '{name}' before confirming.")
                return
            try:
                val = float(text)
            except ValueError:
                QMessageBox.warning(self, "Invalid Number",
                                    f"Could not parse '{text}' as a number for '{name}'.")
                return
            # Enforce bounds
            lo = float(self.param_bounds[name][0].text())
            hi = float(self.param_bounds[name][1].text())
            if not (lo < val < hi):
                QMessageBox.warning(self, "Out of Bounds",
                                    f"Value for '{name}' must be between {lo} and {hi}.")
                return
            params[name] = val
        self.confirmed_params = params
        self.confirm_btn.setEnabled(False)
        self.log_activity(f"Parameters confirmed: {params}")
        # Notify worker via signal/event
        if hasattr(self, '_pending_worker') and self._pending_worker:
            self._pending_worker.on_param_confirmed(params)
            self._pending_worker = None

    def update_pareto_plot(self):
        """Update the Pareto plot with current self.pareto_data."""
        import numpy as np
        if not self.pareto_data:
            self.pareto_scatter.setData([], [])
            if self.pareto_colorbar:
                self.pareto_colorbar.hide()
            return
        specs = np.array([d['spec'] for d in self.pareto_data])
        charges = np.array([d['charge'] for d in self.pareto_data])
        stabs = np.array([d['stability'] for d in self.pareto_data])
        # Normalize stability for color mapping
        stab_norm = (stabs - stabs.min()) / (stabs.ptp() if stabs.ptp() else 1)
        colors = [pg.intColor(int(s*255), 255, alpha=200) for s in stab_norm]
        self.pareto_scatter.setData(specs, charges, symbol='o', symbolBrush=colors, symbolSize=12)
        # Add/update colorbar
        if self.pareto_colorbar is None:
            self.pareto_colorbar = pg.ColorBarItem(values=(stabs.min(), stabs.max()), cmap='viridis')
            self.pareto_colorbar.setImageItem(self.pareto_scatter)
            self.pareto_colorbar.setLevels((stabs.min(), stabs.max()))
            self.pareto_colorbar.setParentItem(self.pareto_plot.getPlotItem())
        else:
            self.pareto_colorbar.setLevels((stabs.min(), stabs.max()))
            self.pareto_colorbar.show()

    def reset_plots_and_analysis(self):
        """Clear all plots and analysis data, including Pareto plot."""
        for hist in self.history.values():
            hist.clear()
        for c in self.curves.values():
            c.setData([], [])
        self.shot_number = 0
        self.pareto_data.clear()
        self.update_pareto_plot()
        self.log_activity("Plots and analysis reset.")

    def _on_pareto_click(self, event):
        """Handle mouse click on Pareto plot: show params for nearest point."""
        import numpy as np
        if not self.pareto_data:
            return
        mouse_point = self.pareto_plot.getPlotItem().vb.mapSceneToView(event.scenePos())
        mx, my = mouse_point.x(), mouse_point.y()
        specs = np.array([d['spec'] for d in self.pareto_data])
        charges = np.array([d['charge'] for d in self.pareto_data])
        dists = np.hypot(specs - mx, charges - my)
        idx = np.argmin(dists)
        if dists[idx] < 0.1 * max(1, np.ptp(specs), np.ptp(charges)):  # Only if close enough
            params = self.pareto_data[idx].get('params', {})
            msg = f"Pareto point:\nSpec: {specs[idx]:.3g}, Charge: {charges[idx]:.3g}\nStability: {self.pareto_data[idx]['stability']:.3g}\nParams: {params}"
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Pareto Point Details", msg)

    def show_current_parameters(self):
        """
        Scan the watched folders for the latest output files and display averaged metrics
        (spec, charge, stability, overall) for the last N shots in the activity log.
        Also plot these metrics on the three plots.
        """
        import numpy as np
        N = self.shots_spin.value()
        mode = self.get_mode()
        scores, charges = [], []
        if mode == 'E-Spec' and hasattr(self, 'espec_dir'):
            path = self.espec_dir
            txt_files = sorted(glob.glob(os.path.join(path, '*.txt')))
            csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))
            all_files = txt_files + csv_files
            all_files = sorted(all_files)
            for fn in all_files:
                try:
                    if fn.endswith('.txt'):
                        line = open(fn, 'r').readline().strip()
                        sc, cost, ct = map(float, line.split(','))
                        scores.append(sc)
                        charges.append(ct)
                    elif fn.endswith('.csv'):
                        score_val = None
                        charge_val = None
                        with open(fn, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith('#'):
                                    continue
                                parts = line.split(',')
                                if len(parts) != 2:
                                    continue
                                key, value = parts[0].strip().lower(), parts[1].strip()
                                if key == 'score':
                                    try:
                                        score_val = float(value)
                                    except Exception:
                                        pass
                                elif key == 'raw':
                                    try:
                                        charge_val = float(value)
                                    except Exception:
                                        pass
                        if score_val is not None:
                            scores.append(score_val)
                        if charge_val is not None:
                            charges.append(charge_val)
                except Exception:
                    pass
        elif mode == 'Profile & ICT' and hasattr(self, 'profile_dir') and hasattr(self, 'ict_dir'):
            prof_files = sorted(glob.glob(os.path.join(self.profile_dir, '*.txt')))
            ict_files = sorted(glob.glob(os.path.join(self.ict_dir, '*.txt')))
            try:
                scores = [float(open(f).readline()) for f in prof_files]
            except Exception:
                scores = []
            try:
                charges = [float(open(f).readline()) for f in ict_files]
            except Exception:
                charges = []
        # Grouped averaging and plotting
        def grouped_average(arr, window):
            arr = np.array(arr)
            n_groups = len(arr) // window
            if n_groups == 0:
                return [], []
            means = [float(np.mean(arr[i*window:(i+1)*window])) for i in range(n_groups)]
            stds = [float(np.std(arr[i*window:(i+1)*window])) for i in range(n_groups)]
            return means, stds
        if scores and charges:
            avg_scores, std_scores = grouped_average(scores, N)
            avg_charges, std_charges = grouped_average(charges, N)
            avg_overall = [float(np.sqrt(s * c)) if s > 0 and c > 0 else 0.0 for s, c in zip(avg_scores, avg_charges)]
            shots = list(range(len(avg_scores)))
            self.plot_spec.clear()
            self.plot_spec.plot(shots, avg_scores, pen='b', symbol='o')
            self.plot_charge.clear()
            self.plot_charge.plot(shots, avg_charges, pen='r', symbol='o')
            self.plot_overall.clear()
            self.plot_overall.plot(shots, avg_overall, pen='g', symbol='o')

    def load_espec_results_csv(self, file_path=None):
        """Load E-Spec results from CSV, average over last N rows, and plot diagnostic score and charge."""
        import pandas as pd
        import os
        if file_path is None:
            # Try to find the file in espec_dir
            if hasattr(self, 'espec_dir'):
                # Look for a file ending with .csv
                files = [f for f in os.listdir(self.espec_dir) if f.endswith('.csv')]
                if files:
                    file_path = os.path.join(self.espec_dir, files[0])
                else:
                    self.log_activity("No CSV file found in E-Spec directory.")
                    return
            else:
                self.log_activity("E-Spec directory not set.")
                return
        try:
            df = pd.read_csv(file_path)
            # Try to find columns for score and charge
            score_col = None
            charge_col = None
            for col in df.columns:
                if 'score' in col.lower():
                    score_col = col
                if 'charge' in col.lower() or 'counts' in col.lower():
                    charge_col = col
            if score_col is None or charge_col is None:
                # Fallback: use first and second columns
                score_col = df.columns[0]
                charge_col = df.columns[1]
            N = self.shots_spin.value()
            scores = df[score_col].values[-N:]
            charges = df[charge_col].values[-N:]
            shots = range(len(df) - N + 1, len(df) + 1)
            # Plot Diagnostic Score
            self.plot_spec.clear()
            self.plot_spec.plot(list(shots), list(scores), pen='b', symbol='o')
            # Plot Charge Metric
            self.plot_charge.clear()
            self.plot_charge.plot(list(shots), list(charges), pen='r', symbol='o')
            # Log average values
            avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
            avg_charge = sum(charges) / len(charges) if len(charges) > 0 else 0
            self.log_activity(f"Loaded {N} E-Spec results from {os.path.basename(file_path)}. Avg score: {avg_score:.3f}, Avg charge: {avg_charge:.3f}")
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "E-Spec Load Error", str(e))
            self.log_activity(f"Error loading E-Spec CSV: {e}")

if __name__=='__main__':
    app = QApplication(sys.argv)

    # Font configuration: Helvetica, size 10, italic
    font = QFont("Roboto", 10)
    # font.setItalic(True)
    app.setFont(font)

    db = QFontDatabase()
    # for family in db.families():
    #     print(family)

    gui = LPAControlGUI()
    gui.show()
    sys.exit(app.exec_())