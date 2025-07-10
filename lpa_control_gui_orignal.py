# lpa_control_gui.py
# PyQt5-based control GUI for LPA Bayesian Optimization
# Includes threading, efficient plotting, file-watcher placeholders,
# error handling, persistent logging, mode switching, and full integration.

import sys              # System-specific parameters and functions
import os               # Miscellaneous operating system interfaces
import glob             # Unix style pathname pattern expansion
import time             # Time access and conversions
import csv              # CSV file reading and writing
from threading import Thread, Event

from bo_engine import compute_objective, create_xopt  # Custom Bayesian Optimization engine

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
from PyQt5.QtCore import Qt, pyqtSignal, QObject  # Core Qt enums, custom signals, base object
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

class OptimizationWorker(Thread):
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
        self.gui = gui                    # GUI object to talk to buttons/fields
        self.signals = WorkerSignals()    # Custom signals back to the GUI thread
        self.stop_event = stop_event      # External flag to stop this loop

        # Prepare a CSV log file in the working folder
        self.log_file = os.path.join(os.getcwd(), 'params_metrics_log.csv')
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['shot','params','overall','spec','charge','timestamp'])

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

        # Main loop: keep going until user stops
        while not self.stop_event.is_set():
            shot += 1

            # Generate new suggestions using Bayesian optimization
            step = self.X.step()
            suggestions = dict(step.data.iloc[-1])

            # Tell GUI: here are the next settings to try
            self.signals.new_suggestions.emit(suggestions)

            # Wait until GUI reports you’ve applied them and clicked “Confirm”
            self.gui.confirmed_params = suggestions.copy()
            while self.gui.confirmed_params is not None and not self.stop_event.is_set():
                time.sleep(0.1)
            if self.stop_event.is_set():
                break

            # Grab the confirmed settings and clear that flag
            params = self.gui.confirmed_params.copy()
            self.gui.confirmed_params = None

            # TODO: actually push these settings to your hardware

            # Read back the last few shots’ data (up to a timeout)
            metrics = self.collect_metrics(timeout=5)
            if metrics is None:
                # If data didn’t arrive, mark as missing
                metrics = {'overall': None, 'spec': None, 'charge': None}

            # Calculate the scalar objective from diagnostics
            score = compute_objective(metrics)
            metrics['overall'] = score

            # Send the metrics to the GUI for real-time plotting
            self.signals.new_metrics.emit(metrics)

            # Append this shot’s info to the CSV log
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    shot,
                    params,
                    metrics['overall'],
                    metrics['spec'],
                    metrics['charge'],
                    time.time()
                ])

            # Store the evaluated result into the Xopt history
            self.X.add_data(params | {'score': score})

        # Signal that we’ve fully stopped
        self.signals.finished.emit()

    def evaluate(self, inputs_list):
        """
        Receives suggestions from Xopt, pushes them through GUI + measurement + scoring,
        and returns the computed score. Called internally by Xopt.
        """
        results = []
        for params in inputs_list:
            # 1. Feed the suggested parameters into the GUI’s confirmed_params
            self.gui.confirmed_params = params.copy()
            # 2. Wait until the GUI thread picks up these params and clears them
            #    (indicates the shot has been triggered/applied)
            while self.gui.confirmed_params is not None and not self.stop_event.is_set():
                time.sleep(0.1)
            # 3. If a stop was requested, exit the loop early
            if self.stop_event.is_set():
                break
            # 4. Collect measurement metrics (averaged over recent shots)
            metrics = self.collect_metrics(timeout=5)
            # 5. Compute the optimization objective (user‐defined) from raw metrics
            score = compute_objective(metrics)
            # 6. Store the result in the form Xopt expects
            results.append({'score': score})
        return results

    def collect_metrics(self, timeout=5):
        """
        Look in the measurement folders for your latest E-Spec or Profile & ICT data.
        - In E-Spec mode: read score,cost,counts from espectra .txt files.
        - In Profile & ICT mode: fallback to spec from profile_dir and charge from ict_dir.
        Returns a dict when both spec & charge are ready, or None after timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            mode = self.gui.get_mode()  # "E-Spec" or "Profile & ICT"
            overall = 1.0  # placeholder, overwritten by compute_objective()

            spec = None
            charge = None

            N = self.gui.shots_spin.value()

            if mode == 'E-Spec':
                # ----- E-Spec branch -----
                path = self.gui.espec_dir
                files = sorted(glob.glob(os.path.join(path, '*.txt')))
                if len(files) >= N:
                    scores, counts = [], []
                    for fn in files[-N:]:
                        try:
                            line = open(fn, 'r').readline().strip()
                            sc, cost, ct = map(float, line.split(','))  # score,cost,counts
                            scores.append(sc)
                            counts.append(ct)
                        except Exception as e:
                            print(f"Error parsing E-Spec file {fn}: {e}")
                    if scores:
                        spec   = sum(scores) / len(scores)
                        charge = sum(counts) / len(counts)
            else:
                # ----- Profile & ICT branch (unchanged) -----
                # spec from profile_dir
                prof = self.gui.profile_dir
                prof_files = sorted(glob.glob(os.path.join(prof, '*.txt')))
                if len(prof_files) >= N:
                    try:
                        vals = [float(open(f).readline()) for f in prof_files[-N:]]
                        spec = sum(vals) / len(vals)
                    except Exception as e:
                        print(f"Error reading profile spec files: {e}")

                # charge from ict_dir
                ict = self.gui.ict_dir
                ict_files = sorted(glob.glob(os.path.join(ict, '*.txt')))
                if len(ict_files) >= N:
                    try:
                        vals = [float(open(f).readline()) for f in ict_files[-N:]]
                        charge = sum(vals) / len(vals)
                    except Exception as e:
                        print(f"Error reading ICT charge files: {e}")

            # Once we have both metrics, return them
            if spec is not None and charge is not None:
                return {'overall': overall, 'spec': spec, 'charge': charge}

            time.sleep(0.1)

        # Timeout: no full data set available
        return

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

        # ---------------------------------------------------------
        # Layout containers
        # ---------------------------------------------------------
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        self.setCentralWidget(w)

        # ---------------------------------------------------------
        # Directory selection buttons (where to read measurement files)
        # ---------------------------------------------------------
        nd = QHBoxLayout()
        nd.setSpacing(20)
        for key, label in [('espec', 'E-Spec'), ('profile', 'Profile'), ('ict', 'ICT')]:
            btn = QPushButton(f"Set {label} Dir")
            # On click, open folder dialog and store path in self.espec_dir, etc.
            btn.clicked.connect(lambda _, k=key: self.choose_folder(k))
            nd.addWidget(btn)
        l.addLayout(nd)

        # ---------------------------------------------------------
        # Mode selector (choose which diagnostics to average)
        # ---------------------------------------------------------
        ml = QHBoxLayout()
        ml.addWidget(QLabel("Diagnostics Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["E-Spec", "Profile & ICT"])
        ml.addWidget(self.mode_combo)
        l.addLayout(ml)

       # ---------------------------------------------------------
        # Manual Explore/Exploit slider (hidden until enabled)
        # ---------------------------------------------------------
        manual_layout = QHBoxLayout()
        self.manual_checkbox = QCheckBox("Manual mode")
        manual_layout.addWidget(self.manual_checkbox)

        # Left label
        self.explore_label_left  = QLabel("Explore")
        self.explore_label_left.setVisible(False)
        manual_layout.addWidget(self.explore_label_left)

        # 3-point slider (0=Explore,1=Balanced,2=Exploit)
        self.explore_slider = QSlider(Qt.Horizontal)
        self.explore_slider.setRange(0, 2)
        self.explore_slider.setValue(1)
        self.explore_slider.setTickPosition(QSlider.TicksBelow)
        self.explore_slider.setTickInterval(1)
        self.explore_slider.setEnabled(False)
        self.explore_slider.setVisible(False)
        # fix initial width to half the window
        self.explore_slider.setFixedWidth(self.width() // 2)
        manual_layout.addWidget(self.explore_slider)

        # Right label
        self.explore_label_right = QLabel("Exploit")
        self.explore_label_right.setVisible(False)
        manual_layout.addWidget(self.explore_label_right)

        l.addLayout(manual_layout)

        # Toggle slider + labels when checkbox is clicked
        self.manual_checkbox.toggled.connect(self._on_manual_toggled)

        # ---------------------------------------------------------
        # Parameter activation checkboxes (which knobs to vary)
        # ---------------------------------------------------------
        pl = QHBoxLayout()
        pl.addWidget(QLabel("Select active parameters:"))
        self.param_checks = {}
        for name in ['plasma_z', 'gas_pressure', 'GDD_phi2', 'blade_x', 'blade_z']:
            cb = QCheckBox(name)
            cb.setChecked(True)
            pl.addWidget(cb)
            self.param_checks[name] = cb
        l.addLayout(pl)

        # ---------------------------------------------------------
        # Control buttons: start, pause, restart optimization
        # ---------------------------------------------------------
        cl = QHBoxLayout()
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self.start_optimization)
        self.pause_btn = QPushButton("Pause Optimization")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_optimization)
        self.restart_btn = QPushButton("Restart Optimization")
        self.restart_btn.clicked.connect(self.restart_optimization)
        self.copy_btn = QPushButton("Copy Suggestions to Use")
        self.copy_btn.clicked.connect(self.copy_suggestions_to_use)
        self.confirm_btn = QPushButton("Confirm Parameters")
        self.confirm_btn.clicked.connect(self.confirm_parameters)
        self.confirm_btn.setEnabled(False)  # only enabled once suggestions are displayed

        for btn in (self.start_btn, self.pause_btn, self.restart_btn,
                    self.copy_btn, self.confirm_btn):
            cl.addWidget(btn)
        l.addLayout(cl)

        # ---------------------------------------------------------
        # Parameter table: [Lower] [Upper] [Parameter] [Suggest] [Use]
        # ---------------------------------------------------------
        grid = QGridLayout()
        cols = ["Lower", "Upper", "Parameter", "Suggest", "Use"]
        for c, title in enumerate(cols):
            lbl = QLabel(title)
            lbl.setStyleSheet("font-weight:bold")
            grid.addWidget(lbl, 0, c)

        self.param_widgets = {}
        self.param_bounds  = {}
        param_list = ['plasma_z', 'gas_pressure', 'GDD_phi2', 'blade_x', 'blade_z']
        for r, name in enumerate(param_list, start=1):
            # Lower bound box
            lo = QLineEdit("0.0")
            lo.setFixedWidth(100)
            grid.addWidget(lo, r, 0)

            # Upper bound box
            hi = QLineEdit("1.0")
            hi.setFixedWidth(100)
            grid.addWidget(hi, r, 1)

            # Parameter name
            grid.addWidget(QLabel(name), r, 2)

            # Suggested value (read-only)
            sug = QLineEdit()
            sug.setReadOnly(True)
            grid.addWidget(sug, r, 3)

            # “Use” value
            val = QLineEdit()
            val.editingFinished.connect(
                lambda _, n=name, w=val: self._confirm_param(n, w)
            )
            grid.addWidget(val, r, 4)

            # Store for later
            self.param_bounds[name]  = (lo, hi)
            self.param_widgets[name] = (sug, val)

        l.addLayout(grid)

        # ---------------------------------------------------------
        # Spin box: number of recent shots to average for diagnostics
        # ---------------------------------------------------------
        sl = QHBoxLayout()
        sl.addWidget(QLabel("Shots to average:"))
        self.shots_spin = QSpinBox()
        self.shots_spin.setMinimum(1)
        self.shots_spin.setValue(10)  # Changed default from 4 to 10
        sl.addWidget(self.shots_spin)
        l.addLayout(sl)

        # ---------------------------------------------------------
        # Plot widgets for live-updating metrics
        # ---------------------------------------------------------
        self.plot_overall = pg.PlotWidget(title="Overall Objective")
        self.plot_spec    = pg.PlotWidget(title="Diagnostic Score")
        self.plot_charge  = pg.PlotWidget(title="Charge Metric")
        for p in [self.plot_overall, self.plot_spec, self.plot_charge]:
            # light grid to guide the eye
            p.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
            l.addWidget(p)

        # Pre-create curves so we can efficiently append new data points
        pen = pg.mkPen('#00796b', width=2)
        brush = '#009688'
        self.curves = {
            'overall': self.plot_overall.plot(pen=pen, symbol='o', symbolBrush=brush),
            'spec':    self.plot_spec.plot(pen=pen, symbol='o', symbolBrush=brush),
            'charge':  self.plot_charge.plot(pen=pen, symbol='o', symbolBrush=brush),
        }

    def resizeEvent(self, ev):
            """Keep slider at half the window width whenever the window is resized."""
            super().resizeEvent(ev)
            # only adjust if the slider exists
            if hasattr(self, 'explore_slider'):
                self.explore_slider.setFixedWidth(self.width() // 2)

    def copy_suggestions_to_use(self):
        """Copy the suggested values into the 'Use' input boxes."""
        for name, (sug, use) in self.param_widgets.items():
            use.setText(sug.text())

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
            setattr(self, f"{key}_dir", path)
            QMessageBox.information(self, "Path Set", f"{key} dir: {path}")

    def start_optimization(self):
        """Kick off the background OptimizationWorker thread."""
        self.stop_event.clear()
        worker = OptimizationWorker(self, self.stop_event)
        # Hook up signals to update GUI when new data arrives
        worker.signals.new_metrics.connect(self.record_and_plot)
        worker.signals.new_suggestions.connect(self.update_suggestions)
        worker.start()
        self.worker = worker
        # Update button states
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.restart_btn.setEnabled(False)

    def pause_optimization(self):
        """Temporarily pause sending new suggestions to hardware."""
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)

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
        return self.mode_combo.currentText()

    def update_suggestions(self, sugg):
        """
        When the worker emits new_suggestions:
         - Clear any previously confirmed values
         - Display the new suggested values in the left-hand boxes
        """
        self.confirmed_params = None
        for name, val in sugg.items():
            if name in self.param_widgets:
                self.param_widgets[name][0].setText(str(val))
        
        # after you clear confirmed_params and update sug.text():
        self.confirm_btn.setEnabled(True)

    def record_and_plot(self, metrics):
        """
        When the worker emits new_metrics:
         - Increment shot counter
         - Append each metric to history
         - Re-draw the curves with the updated data
        """
        self.shot_number += 1
        for key in ['overall', 'spec', 'charge']:
            val = metrics.get(key)
            if val is not None:
                self.history[key].append((self.shot_number, val))
                xs, ys = zip(*self.history[key])
                self.curves[key].setData(xs),

    def confirm_parameters(self):
        """
        Read the current “Use” values and set them as confirmed_params,
        so the OptimizationWorker can proceed.
        """
        params = {}
        for name, (sug, use) in self.param_widgets.items():
            text = use.text().strip()
            if text == "":
                QMessageBox.warning(self, "Missing Value",
                                    f"Please enter a value for '{name}' before confirming.")
                return
            try:
                params[name] = float(text)
            except ValueError:
                QMessageBox.warning(self, "Invalid Number",
                                    f"Could not parse '{text}' as a number for '{name}'.")
                return

        # Store for the worker thread
        self.confirmed_params = params

        # Disable until next suggestion
        self.confirm_btn.setEnabled(False)

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