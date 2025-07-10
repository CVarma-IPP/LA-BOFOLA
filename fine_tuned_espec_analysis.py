from math import cos
import os
import sys
import json
from typing import final
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.interpolate import interp1d
from scipy.integrate import simpson as simps
import easygui as eg
from datetime import datetime
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

# Local modules
import Analysis_code_functions as acf

def load_and_prepare_calibration(calib_path):
    """
    Load calibration data and compute interpolators and transform.

    Parameters
    ----------
    calib_path : str
        Path to the calibration file.

    Returns
    -------
    dict
        Dictionary containing the calibration data and computed values.
    """
    data = acf.load_calibration_file(calib_path)
    corners = [
        [data['Corner_TL_x'], data['Corner_TL_y']],
        [data['Corner_TR_x'], data['Corner_TR_y']],
        [data['Corner_BR_x'], data['Corner_BR_y']],
        [data['Corner_BL_x'], data['Corner_BL_y']],
    ]
    src, dst, maxWid, maxLen = acf.input_for_proj_mat(*corners)
    projmat = acf.calc_perspective_transform_matrix(src, dst)

    ds_dE = np.loadtxt(data['Distance to Energy Calibration File'])
    energy = ds_dE[:, 0]
    dist = ds_dE[:, 1]
    energy_interp = interp1d(dist, energy, kind='linear', fill_value='extrapolate')
    dE_ds_interp = interp1d(dist, np.gradient(energy, dist), kind='slinear')

    return {
        'projmat': projmat,
        'maxWid': maxWid,
        'maxLen': maxLen,
        'pixel2mm': data['Average pixel to mm conversion ratio'],
        'screen_to_axis_distance': data['The distance of the Lanex to the laser beam axis along the screen axis (mm)'],
        'average_distance_covered': data['Average Distance Covered (mm)'],
        'energy_partition': data['Energy partition used for background and divergence calculations (MeV)'],
        'energy_interp': energy_interp,
        'dE_ds_interp': dE_ds_interp,
    }

def aimed_spectrum(energies, peak=100.0, sigma=5, amplitude=1.0):
    """
    Define the target (aimed) spectrum as a normalized Gaussian.

    Parameters
    ----------
    energies : ndarray
        1D array of energy values (MeV).
    peak : float, optional
        Desired peak energy (MeV).
    sigma : float, optional
        Standard deviation of Gaussian (MeV).
    amplitude : float, optional
        Gaussian peak amplitude before normalization.

    Returns
    -------
    ndarray
        Normalized Gaussian counts.
    """
    target_counts = amplitude * np.exp(-0.5 * ((energies - peak) / sigma)**2)
    if np.max(target_counts) > 0:
        target_counts /= np.max(target_counts)
    return target_counts

def compute_fwhm(x, y):
    half = y.max() / 2.0
    inds = np.where(y >= half)[0]
    if inds.size < 2: return 0.0
    left, right = inds[0], inds[-1]
    def interp(i0, i1):
        x0, x1 = x[i0], x[i1]
        y0, y1 = y[i0], y[i1]
        return x0 if y1==y0 else x0 + (half-y0)*(x1-x0)/(y1-y0)
    eL = interp(left-1,left) if left>0 else x[0]
    eR = interp(right,right+1) if right<len(y)-1 else x[-1]
    return eR - eL

def smooth_savgol(counts, energies, window_MeV=1.0, polyorder=2):
    """
    Smooth the counts using a Savitzky-Golay filter.

    Parameters
    ----------
    counts : ndarray
        1D array of counts corresponding to the energies.
    energies : ndarray
        1D array of energy values (MeV).
    window_MeV : float, optional
        Width of the smoothing window in MeV.
    polyorder : int, optional
        Order of the polynomial used in the Savitzky-Golay filter.

    Returns
    -------
    ndarray
        Smoothed counts.    
    """

    ΔE = energies[1] - energies[0]
    window_bins = int(round(window_MeV / ΔE))
    # window must be odd and ≥ polyorder+2
    if window_bins % 2 == 0:
        window_bins += 1
    window_bins = max(window_bins, polyorder + 2)
    return savgol_filter(counts, window_length=window_bins, polyorder=polyorder)

def evaluate_spectrum_v3(exp_spectrum, target_spectrum_func, *, region=(50,200), low_energy_threshold=50,
                         target_peak=100.0, overshoot_floor=0.1, weights=None, worst_cost=100.0):
    """
    Evaluate the quality of an electron energy spectrum compared to a target distribution.
    
    Computes a weighted quality score by comparing an experimental electron spectrum to a 
    target distribution (typically Gaussian). The evaluation considers seven quality metrics:
    shape matching, peak position, energy spread, kurtosis (peakedness), presence of multiple 
    peaks, low-energy content, and high-energy overshoot. Returns a normalized score (0-100)
    and individual component contributions.
    
    Parameters
    ----------
    exp_spectrum : ndarray, shape (n, 2)
        Experimental spectrum as a 2D array with columns [energy, counts].
    target_spectrum_func : callable
        Function that takes an energy array and returns the target spectrum counts.
        Should match the signature of `aimed_spectrum()`.
    region : tuple of float, optional
        Energy range (min, max) in MeV for the main analysis region.
    low_energy_threshold : float, optional
        Energy threshold (MeV) below which electrons are considered "low energy"
        and penalized.
    target_peak : float, optional
        Target peak energy (MeV) for the ideal spectrum.
    overshoot_floor : float, optional
        Minimum normalized intensity to consider for the overshoot bonus calculation.
    weights : dict, optional
        Weighting factors for each penalty component. Must contain keys:
        'shape', 'peak_pos', 'spread', 'kurtosis', 'multipeak', 'low_energy', 'overshoot'.
        Negative weights convert penalties to bonuses.
    worst_cost : float, optional
        Maximum cost value used to normalize the final score to a 0-100 scale.
        
    Returns
    -------
    dict
        Dictionary containing the overall score and individual penalty components:
        - 'score': Overall quality score (0-100, higher is better)
        - 'raw': Raw weighted sum of all penalty terms
        - 'shape': Weighted penalty for shape deviation
        - 'peak': Weighted penalty for peak position deviation
        - 'spread': Weighted penalty for energy spread deviation
        - 'kurtosis': Weighted penalty for kurtosis (peakedness) deviation
        - 'multipeak': Weighted penalty for multiple peaks
        - 'low': Weighted penalty for low-energy content
        - 'over': Weighted bonus for high-energy overshoot
    
    Notes
    -----
    Negative weights convert penalties into bonuses. For example, a negative 'low_energy'
    weight would reward spectra with significant low-energy content, while a negative
    'overshoot' weight (default behavior) rewards spectra with peak energies exceeding
    the target peak.
    
    A typical weights dictionary for VHEE applications might be:
    weights = {
        'shape': 5.0,       # Strong emphasis on overall distribution shape
        'peak_pos': 0.5,    # Moderate emphasis on peak position
        'spread': 0.05,     # Low emphasis on exact energy spread
        'low_energy': -2.0, # Moderate penalty for low-energy content
        'kurtosis': 0.05,   # Low emphasis on exact peakedness
        'multipeak': 5.0,   # Strong penalty for multiple peaks
        'overshoot': 1.0    # Moderate bonus for higher-than-target energy
    }
    """
    energies = exp_spectrum[:,0]
    counts   = exp_spectrum[:,1]

    # Use the filter to smooth the counts before evaluation
    counts = smooth_savgol(counts, energies, window_MeV=5, polyorder=2)

    if counts.max() <= 0:
        return {k: 0.0 for k in
            ('score','raw','shape','peak','spread','kurtosis','multipeak','low','over')}
    norm_counts = counts / counts.max()

    target = target_spectrum_func(energies)
    if target.max() > 0:
        target = target / target.max()

    # Mask to region, once
    lo, hi = region
    inreg = (energies >= lo) & (energies <= hi)
    if not inreg.any():
        return {k:0.0 for k in
            ('score','raw','shape','peak','spread','kurtosis','multipeak','low','over')}

    em = energies[inreg]        # energy in region
    nc = norm_counts[inreg]     # normalized counts in region     
    tg = target[inreg]          # target counts in region

    # 1) shape
    w_gauss     = np.exp(-0.5 * ((em - target_peak)/((hi-lo)/4))**2)    # Gaussian weights
    err         = nc - tg                                               # error in region
    shape_pen   = (w_gauss*err**2).sum() / w_gauss.sum()                # weighted MSE
    shape_c     = weights['shape'] * shape_pen                          

    # 2) peak position
    e_exp       = em[np.argmax(nc)]                                     # energy of the peak in exp spectrum
    e_tgt       = em[np.argmax(tg)] or 1.0                              # energy of the peak in target spectrum               
    peak_pen    = abs(e_exp - e_tgt)/e_tgt                              # relative difference in peak position
    peak_c      = weights['peak_pos'] * peak_pen        

    # 3) spread 
    fwhm_val    = compute_fwhm(em, nc)                                  # FWHM of the experimental spectrum
    ideal_fwhm  = 2*np.sqrt(2*np.log(2))*((hi-lo)/4)                    # Ideal FWHM for a Gaussian in the region
    spread_pen  = abs((fwhm_val/(e_exp or 1.0)) - (ideal_fwhm/target_peak)) # relative difference in FWHM
    spread_c    = weights['spread'] * spread_pen

    # 4) kurtosis
    kurt_val    = abs(kurtosis(nc, fisher=False) - kurtosis(tg, fisher=False))  # absolute difference in kurtosis
    kurt_c      = weights['kurtosis'] * kurt_val

    # 5) multipeak
    peaks, _   = find_peaks(nc, height=0.1)                             # find peaks in the normalized counts
    multi_pen   = max(0, len(peaks)-1)/nc.size                          # penalty for multiple peaks (1 if more than 1 peak) 
    multi_c     = weights['multipeak'] * multi_pen

    # 6) low-energy
    lowmask   = energies < low_energy_threshold                         # mask for low-energy region
    low_area  = np.trapz(norm_counts[lowmask], x=energies[lowmask]) if lowmask.any() else 0.0 # area under the low-energy counts
    low_c     = weights['low_energy'] * low_area

    # 7) overshoot
    overmask  = (energies > target_peak) & (norm_counts > overshoot_floor)  # mask for overshoot region
    over_val  = norm_counts[overmask].sum() if overmask.any() else 0.0      # total counts in the overshoot region
    over_c    = -weights['overshoot'] * over_val

    # Combine all costs and normalize to score
    raw = shape_c + peak_c + spread_c + kurt_c + multi_c + low_c + over_c
    clamped = min(max(raw, 0.0), worst_cost)
    score   = 100.0 * (1.0 - clamped / worst_cost)

    return {
        'score':   score,
        'raw':     raw,
        'shape':   shape_c,
        'peak':    peak_c,
        'spread':  spread_c,
        'kurtosis':kurt_c,
        'multipeak':multi_c,
        'low':     low_c,
        'over':    over_c
    }

def interactive_preview(img, calib, mode, roi, fine):
    """
    Launch an interactive preview with adjustable ROI threshold, energy range,
    and exclusion of multiple burnt‐pixel regions.

    Returns
    -------
    emin : float
        Lower bound of energy range after adjustment.
    emax : float
        Upper bound of energy range after adjustment.
    new_roi : float
        Final ROI selection percentage.
    burn_ranges : list of (float,float)
        List of user‐entered “burnt” intervals [(lo1, hi1), (lo2, hi2), ...].
    """
    import numpy as np
    from matplotlib.widgets import Slider, TextBox
    from matplotlib.ticker import FormatStrFormatter

    # 1) Show the “raw” processed image once for reference
    acf.process_image(
        img, calib['maxWid'], calib['maxLen'], calib['projmat'],
        calib['pixel2mm'], calib['screen_to_axis_distance'], calib['dE_ds_interp'],
        calib['average_distance_covered'], calib['energy_interp'], calib['energy_partition'],
        ROI_selection_percent=roi, fine_cut_flag=fine,
        bg_flag=True, PT_flag=True, norm_flag=False,
        plot_flag=True)

    # 2) Compute the spectrum once (no plotting flag)
    intensity_energy, x_axis, _, _ = acf.process_image(
        img, calib['maxWid'], calib['maxLen'], calib['projmat'],
        calib['pixel2mm'], calib['screen_to_axis_distance'], calib['dE_ds_interp'],
        calib['average_distance_covered'], calib['energy_interp'], calib['energy_partition'],
        ROI_selection_percent=roi, fine_cut_flag=fine,
        bg_flag=True, PT_flag=True, norm_flag=False,
        plot_flag=False)

    # Initialize the parameters we will return
    emin, emax, new_roi = float(x_axis.min()), float(x_axis.max()), roi
    burn_ranges = []              # list of (lo, hi) that user types in
    final_mask = np.ones_like(x_axis, dtype=bool)  # initially no bins are “burnt”

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(left=0.3, bottom=0.3)

    # Plot the initial line (no masking yet)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    line, = ax.plot(x_axis, intensity_energy, color='C0')
    ax.set(xlabel='Energy (MeV)', ylabel='Counts', title='Interactive Spectrum')

    # 3) Add the ROI slider
    ax_roi = plt.axes([0.3, 0.2, 0.6, 0.03])
    slider_roi = Slider(ax_roi, 'ROI', 0.1, 1.0, valinit=roi, valstep=0.005)

    # 4) Add text boxes for E_min / E_max
    ax_min = plt.axes([0.1, 0.1, 0.1, 0.05])
    txt_min = TextBox(ax_min, 'E_min', initial=f"{emin:.2f}")
    ax_max = plt.axes([0.1, 0.02, 0.1, 0.05])
    txt_max = TextBox(ax_max, 'E_max', initial=f"{emax:.2f}")

    # 5) Add text box for “burnt” intervals
    ax_burn = plt.axes([0.3, 0.15, 0.6, 0.05])
    txt_burn = TextBox(ax_burn, 'Burnt (a-b,...)', initial='')

    def update(val=None):
        nonlocal emin, emax, new_roi, burn_ranges, final_mask
        new_roi = slider_roi.val
        # Parse emin/emax (if invalid, leave them as before)
        try:
            emin = float(txt_min.text)
            emax = float(txt_max.text)
        except ValueError:
            pass

        # Parse the “burnt” text into intervals
        text = txt_burn.text.strip()
        burn_ranges = []
        if text:
            parts = [r.strip() for r in text.split(',')]
            for part in parts:
                lo_hi = part.split('-')
                if len(lo_hi) == 2:
                    try:
                        bl = float(lo_hi[0]); bh = float(lo_hi[1])
                        burn_ranges.append((bl, bh))
                    except ValueError:
                        continue

        # Recompute the spectrum with updated ROI but not plotting
        intensity2, x2, _, _ = acf.process_image(img, calib['maxWid'], calib['maxLen'], calib['projmat'], 
                                                 calib['pixel2mm'], calib['screen_to_axis_distance'], 
                                                 calib['dE_ds_interp'], calib['average_distance_covered'], 
                                                 calib['energy_interp'], calib['energy_partition'], 
                                                ROI_selection_percent=new_roi, fine_cut_flag=fine,
                                                bg_flag=True, PT_flag=True, norm_flag=False,
                                                plot_flag=False)

        # Create boolean masks
        window_mask = (x2 >= emin) & (x2 <= emax)
        burn_mask = np.zeros_like(x2, dtype=bool)
        for (bl, bh) in burn_ranges:
            burn_mask |= (x2 >= bl) & (x2 <= bh)

        # Final mask is “in our chosen window” AND “not in any burnt range”
        final_mask = window_mask & (~burn_mask)

        # Update the line to show only bins that survive masking
        line.set_data(x2[final_mask], intensity2[final_mask])
        ax.set_xlim(emin, emax)
        if np.any(intensity2[final_mask]):
            vmin = intensity2[final_mask].min() * 0.9
            vmax = intensity2[final_mask].max() * 1.1
            ax.set_ylim(vmin, vmax)

        # Reformat x-axis ticks
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.canvas.draw_idle()

    # Register callbacks
    slider_roi.on_changed(update)
    txt_min.on_submit(update)
    txt_max.on_submit(update)
    txt_burn.on_submit(update)

    plt.show()

    # Return the final user‐chosen values
    return emin, emax, new_roi, burn_ranges

def process_batch(files, calib, args):
    """
    Process TIFF images to compute and save two mean spectra:
      1. Using all spectra
      2. Using the top 50% best-matching spectra by Gaussian cost

    Parameters
    ----------
    files : list of str
        Paths to the image files to analyze.
    calib : dict
        Calibration data dict returned by load_and_prepare_calibration, including projection matrix,
        pixel-to-mm conversion, and interpolators for energy and dE/ds.
    args : dict
        Analysis parameters:
        - roi : float
            ROI selection percentage (0 < roi <= 1).
        - fine : bool
            If True, perform fine symmetric cuts for background subtraction.
        - vhee_threshold : float
            Lower energy bound (MeV) for VHEE area integration.
        - emin : float
            Lower energy bound (MeV) for mean/std computation.
        - emax : float
            Upper energy bound (MeV) for mean/std computation.
        - burn_ranges : list of tuple
            List of (low, high) intervals to exclude (“burnt pixels”).
        - format : list of str
            Output image formats (e.g., ['png','svg']).
        - outdir : str
            Directory to save figures and data files.

    Returns
    -------
    None

    Notes
    -----
    - Saves timestamped mean spectrum plots (within [emin, emax]) to outdir in each specified format.
    - Saves corresponding text files with three columns: Energy (MeV), Mean counts, Std deviation.
    - Prints a confirmation message upon successful write.
    - After saving, prints the cost threshold used for selection.
    """
    spectra = []
    costs = []
    energies = None
    burn_mask_global = None
    comparison = True  # Set to True to enable spectrum comparison plots

    # Compute each spectrum and its weighted-cost breakdown
    for f in files:
        img = imageio.imread(f).astype(np.float32)
        intensity, e, _, _ = acf.process_image(img, calib['maxWid'], calib['maxLen'], calib['projmat'],
                                                calib['pixel2mm'], calib['screen_to_axis_distance'], 
                                                calib['dE_ds_interp'], calib['average_distance_covered'], 
                                                calib['energy_interp'], calib['energy_partition'],
                                                ROI_selection_percent=args['roi'], fine_cut_flag=args['fine'],
                                                bg_flag=True, PT_flag=True, norm_flag=False, plot_flag=False)
                                                
        if energies is None:
            energies = e
            # Build a global burn mask from args['burn_ranges']
            burn_ranges = args.get('burn_ranges', [])
            burn_mask_global = np.zeros_like(energies, dtype=bool)
            for (bl, bh) in burn_ranges:
                burn_mask_global |= (energies >= bl) & (energies <= bh)

        spectra.append(intensity)

        # Filter out burnt intervals before cost evaluation
        valid_mask = ~burn_mask_global
        e_filt = e[valid_mask]
        intensity_filt = intensity[valid_mask]
        exp_spec = np.column_stack((e_filt, intensity_filt))

        # Compute weighted cost and all contributions via v3
        result = evaluate_spectrum_v3(exp_spec, aimed_spectrum, region=(args['emin'], args['emax']),
                                        target_peak=100, worst_cost=70, weights={'shape':     35.0,
                                                                                'peak_pos':   15.0,
                                                                                'spread':     5.0,
                                                                                'low_energy': -1.5,
                                                                                'kurtosis':   0.1,
                                                                                'multipeak':  30.0,
                                                                                'overshoot':  1.5
                                                                            })            
        score   = result['score']
        raw_cost = result['raw']
        shape_c  = result['shape']
        peak_c   = result['peak']
        spread_c = result['spread']
        kurt_c   = result['kurtosis']
        multi_c  = result['multipeak']
        low_c    = result['low']
        over_c   = result['over']
        costs.append(raw_cost)

        # Plot normalized comparison with full breakdown
        if comparison:
            lo, hi = args['emin'], args['emax']
            mask_cmp = (e >= lo) & (e <= hi) & (~burn_mask_global)
            norm_int = intensity[mask_cmp]
            if norm_int.max() > 0:
                norm_int = norm_int / norm_int.max()
            targ = aimed_spectrum(e[mask_cmp])
            norm_targ = targ / targ.max() if targ.max() > 0 else targ

            # Smoothen the experimental curve to show the plotted comparison
            norm_int = smooth_savgol(norm_int, e[mask_cmp], window_MeV=5.0, polyorder=2)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(e[mask_cmp], norm_int, label='Normalized Experimental')
            ax.plot(e[mask_cmp], norm_targ, '--', label='Normalized Aimed')

            title = (
                f"Comparison for {os.path.basename(f)}\n"
                f"Score: {score:.1f}/100  |  Raw cost: {raw_cost:.3f}\n"
                f"shape={shape_c:.3f}, peak={peak_c:.3f}, \n"
                f"spread={spread_c:.3f}, kurtosis={kurt_c:.3f}, multi={multi_c:.3f}, \n"
                f"low={low_c:.3f}, over={over_c:.3f}"
            )
            ax.set(xlabel='Energy (MeV)', ylabel='Normalized Counts', title=title)
            ax.legend()
            ax.grid(True)
            plt.show()

    costs = np.array(costs)

    # Mask energies to the user-defined window and remove burnt ranges
    mask_window = (energies >= args['emin']) & (energies <= args['emax'])
    total_mask = mask_window & (~burn_mask_global)
    e_sel = energies[total_mask]
    data_all = np.vstack([s[total_mask] for s in spectra])
    mean_all = data_all.mean(axis=0)
    std_all  = data_all.std(axis=0)

    # Determine top 37% by raw cost (lower is better)
    best_threshold = 37  # Percentage to keep
    n = len(files)
    n_select = max(int(n * best_threshold / 100), 1)
    idx_sorted = np.argsort(costs)
    top_idx = idx_sorted[:n_select]
    cost_threshold = costs[top_idx[-1]]
    data_top = np.vstack([spectra[i][total_mask] for i in top_idx])
    mean_top = data_top.mean(axis=0)
    std_top  = data_top.std(axis=0)

    # Update output filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_all = os.path.join(args['outdir'], f"mean_all_{timestamp}")
    base_top = os.path.join(args['outdir'], f"mean_top{best_threshold}_{timestamp}")

    # Plot and save both aggregate spectra
    for label, mean_s, std_s, base in [ ('All spectra', mean_all, std_all, base_all),
                                        (f'Top {best_threshold}% spectra', mean_top, std_top, base_top)]:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(e_sel, mean_s - std_s, mean_s + std_s, alpha=0.3)
        ax.plot(e_sel, mean_s)
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Counts')
        ax.set_title(f"Mean Spectrum ({label}) [{args['emin']:.1f}-{args['emax']:.1f} MeV]")
        ax.grid(True)
        for fmt in args['format']:
            fig.savefig(f"{base}.{fmt}", dpi=300)
        plt.close(fig)

        # Save data to text file
        out_data = np.column_stack((e_sel, mean_s, std_s))
        txtfn = f"{base}.txt"
        np.savetxt(txtfn, out_data, header='Energy (MeV)\tMean counts\tStd dev', fmt='%.6e', delimiter='\t')

    # Report final threshold
    print(f"Results saved in {args['outdir']}")
    print(f"Gaussian cost threshold for top {best_threshold}%: {cost_threshold:.3f}")

def main():
    # # File selection dialogs
    # calib_file = eg.fileopenbox(msg='Select calibration .txt file', title='Calibration File')
    # modes = ['High Energy (HE)', 'Low Energy (LE)', 'Both']
    # mode = eg.choicebox(msg='Choose analysis mode', choices=modes)
    # files = eg.fileopenbox(msg='Select image files (TIFF)', title='Image Files', multiple=True)
    # outdir = eg.diropenbox(msg='Select output folder', title='Output Directory')
    # roi = float(eg.enterbox(msg='ROI selection percentage (0.1-1.0)', default='0.65'))
    # fine = eg.ynbox(msg='Enable fine symmetric cuts?', title='Fine Cuts')
    # vhee = float(eg.enterbox(msg='VHEE threshold energy (MeV)', default='50.0'))
    # fmts = eg.multchoicebox(msg='Choose output image formats', choices=['png','svg'])
    # preview = eg.ynbox(msg='Launch interactive spectrum preview after processing first image?', title='Preview')

    # os.makedirs(outdir, exist_ok=True)
    # calib = load_and_prepare_calibration(calib_file)

    # if preview and files:
    #     img0 = imageio.imread(files[0]).astype(np.float32)
    #     interactive_preview(img0, calib, mode, roi, fine)

    # args = {'roi': roi, 'fine': fine, 'vhee_threshold': vhee, 'format': fmts, 'outdir': outdir}
    # process_batch(files, calib, args)
    
    # File selection dialogs
    calib_file = eg.fileopenbox(msg='Select calibration .txt file', title='Calibration File')
    files = eg.fileopenbox(msg='Select image files (TIFF)', title='Image Files', multiple=True)
    outdir = eg.diropenbox(msg='Select output folder', title='Output Directory')

    # Debug defaults for analysis parameters
    mode = 'High Energy (HE)'  # Default mode
    roi = 0.65                 # Default ROI selection (65%)
    fine = True                # Enable fine symmetric cuts by default
    vhee = 50.0                # Default VHEE threshold energy (MeV)
    fmts = ['png']             # Default image format
    preview = True             # Launch interactive preview by default

    os.makedirs(outdir, exist_ok=True)
    calib = load_and_prepare_calibration(calib_file)

    burn_ranges = [] # Initialize empty list for burnt pixel ranges

    if preview and files:
        img0 = imageio.imread(files[0]).astype(np.float32)
        emin, emax, roi, burn_ranges = interactive_preview(img0, calib, mode, roi, fine)

    args = {'roi': roi, 'fine': fine, 'vhee_threshold': vhee, 'format': fmts, 
            'outdir': outdir, 'emin': emin, 'emax': emax, 'burn_ranges': burn_ranges}
    process_batch(files, calib, args)

if __name__ == '__main__':
    main()
# This script is designed to be run as a standalone program.