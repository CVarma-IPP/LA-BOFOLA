import sys               # For application startup/exit
import os                # For file path manipulations
import time              # To pause briefly while files are written
from datetime import datetime      # To timestamp output files
from collections import deque      # Efficient fixed-length queue for plotting

# PyQt5 widgets and layouts
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QDoubleSpinBox,
    QGroupBox, QFormLayout, QSizePolicy
)
from PyQt5.QtCore import Qt  # For alignment and orientation flags

# Watch a folder for new files
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Your analysis routine for each new ICT data file
import ICT_live_analysis as ict_analysis

# Embed a Matplotlib figure inside Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    """
    A plot area embedded in the Qt window,
    showing “Charge vs. Shot Number” in real time.
    """
    def __init__(self, parent=None, maxshots: int = 50):
        # Create a Matplotlib Figure and attach it
        self.fig = Figure(figsize=(5, 3))
        super().__init__(self.fig)
        self.setParent(parent)

        # One subplot for our line‐plot
        self.ax = self.fig.add_subplot(111)

        # Keep only the last `maxshots` points for performance
        self.maxshots = maxshots
        self.shots = deque(maxlen=maxshots)    # shot indexes
        self.charges = deque(maxlen=maxshots)  # corresponding charges

        # Label axes and show grid lines
        self.ax.set_xlabel('Shot Number')
        self.ax.set_ylabel('Charge (pC)')
        self.ax.grid(True)
        self.fig.tight_layout()

    def update_plot(self, shot_number: int, charge: float):
        """
        Called whenever a new charge measurement arrives.
        - Append the new (shot, charge) pair
        - Redraw the line plot showing only the last N shots
        """
        self.shots.append(shot_number)
        self.charges.append(charge)

        # Clear old drawing and plot updated data
        self.ax.clear()
        self.ax.plot(list(self.shots), list(self.charges), marker='o')

        # Adjust x-axis to exactly cover the recent shots
        if self.shots:
            self.ax.set_xlim(self.shots[0], self.shots[-1])

        # Reapply labels and grid then render
        self.ax.set_xlabel('Shot Number')
        self.ax.set_ylabel('Charge (pC)')
        self.ax.grid(True)
        self.draw()


class NewFileHandler(FileSystemEventHandler):
    """
    Watches a directory for new data files. When one appears:
     1. Reads the new file
     2. Runs your ICT analysis routine
     3. Logs success or error to the text box
     4. Saves a small _analysis_ text file with the computed charge
     5. Tells the plot to update with the new point
    """
    def __init__(self, out_dir: str, log_widget: QTextEdit,
                 plot_canvas: PlotCanvas, param_getter):
        self.out_dir = out_dir                # where to write analysis results
        self.log = log_widget                 # text box for status messages
        self.plot_canvas = plot_canvas        # live-update plot
        self.param_getter = param_getter      # function returning two calibration numbers
        self.shot_counter = 0                 # how many files we’ve processed

    def on_created(self, event):
        # Ignore directories
        if event.is_directory:
            return

        path = event.src_path
        # Only care about data files with these extensions
        if not path.lower().endswith(('.csv', '.txt', '.dat', '.wave')):
            return

        # Brief pause to ensure the file is fully written
        time.sleep(0.1)

        # Pull current calibration values from the GUI and apply them
        val1, val2 = self.param_getter()
        ict_analysis.VAL1 = val1
        ict_analysis.VAL2 = val2

        # Try to analyze the new data file
        try:
            charge = ict_analysis.analyze_file(path)
        except Exception as e:
            # On error, show a red “ERROR” line in the log
            self.log.append(f"[ERROR] analyzing {os.path.basename(path)}: {e}")
            return

        # Successful: increment our shot count
        self.shot_counter += 1

        # Build an output filename with timestamp
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{name}_analysis_{timestamp}.txt"
        out_path = os.path.join(self.out_dir, out_name)

        # Write the analysis result (charge in pC) to that file
        with open(out_path, 'w') as f:
            f.write(f"charge_pC: {charge:.6f}\n")

        # Log success and update the live plot
        self.log.append(f"✔ {base} → {out_name} (Charge: {charge:.2f} pC)")
        self.plot_canvas.update_plot(self.shot_counter, charge)


class MainWindow(QWidget):
    """
    The main application window:
     - Lets you choose an input folder to watch
     - Lets you choose an output folder for results
     - Shows your two calibration spin-boxes
     - Embeds the live plot and the activity log
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live ICT Monitor")
        self.resize(900, 600)

        # Track which folders are selected
        self.in_dir = ''
        self.out_dir = ''

        # --- Calibration controls ---
        params_box = QGroupBox("Calibration Parameters")
        form = QFormLayout()
        self.spin_val1 = QDoubleSpinBox()
        self.spin_val1.setDecimals(6)
        self.spin_val1.setRange(0.0, 100.0)
        self.spin_val1.setValue(ict_analysis.VAL1)
        self.spin_val2 = QDoubleSpinBox()
        self.spin_val2.setDecimals(6)
        self.spin_val2.setRange(0.0, 100.0)
        self.spin_val2.setValue(ict_analysis.VAL2)
        form.addRow("VAL1:", self.spin_val1)
        form.addRow("VAL2:", self.spin_val2)
        params_box.setLayout(form)

        # --- Folder selection ---
        self.btn_in = QPushButton("Select input folder…")
        self.lbl_in = QLabel("<i>No input folder selected</i>")
        self.btn_out = QPushButton("Select output folder…")
        self.lbl_out = QLabel("<i>No output folder selected</i>")

        # --- Activity log (read-only text area) ---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Live plot canvas ---
        self.plot_canvas = PlotCanvas(self, maxshots=50)
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Layout everything vertically ---
        main = QVBoxLayout(self)
        main.addWidget(params_box)
        main.addWidget(self.btn_in)
        main.addWidget(self.lbl_in)
        main.addWidget(self.btn_out)
        main.addWidget(self.lbl_out)
        main.addWidget(self.plot_canvas)
        main.addWidget(QLabel("Activity log:"))
        main.addWidget(self.log)

        # --- Connect button clicks to folder-choosing methods ---
        self.btn_in.clicked.connect(self.choose_input)
        self.btn_out.clicked.connect(self.choose_output)

        # Placeholder for the folder-watch observer
        self.observer = None

    def choose_input(self):
        """Open a dialog to pick the folder where new data appears."""
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.in_dir = folder
            self.lbl_in.setText(folder)
            self.restart_observer()

    def choose_output(self):
        """Open a dialog to pick where to save analysis results."""
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.out_dir = folder
            self.lbl_out.setText(folder)
            self.restart_observer()

    def restart_observer(self):
        """
        Stop any existing folder-watcher, then if both
        input+output folders are set, start watching for new files.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()

        if self.in_dir and self.out_dir:
            handler = NewFileHandler(
                self.out_dir,
                self.log,
                self.plot_canvas,
                # param_getter returns the two spinbox values
                param_getter=lambda: (self.spin_val1.value(), self.spin_val2.value())
            )
            self.observer = Observer()
            self.observer.schedule(handler, self.in_dir, recursive=False)
            self.observer.start()
            self.log.append(f"▶ Watching {self.in_dir}\n  saving analyses to {self.out_dir}")

    def closeEvent(self, event):
        """Cleanly stop the folder-watcher when the window closes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        super().closeEvent(event)


if __name__ == '__main__':
    # Standard Qt application bootstrap
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())