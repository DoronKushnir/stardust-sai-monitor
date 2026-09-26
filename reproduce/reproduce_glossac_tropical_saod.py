import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # repo root -> saimon importable
import matplotlib.pyplot as plt
import numpy as np
from saimon.glossac import GloSSACLoader, compute_tropical_saod
import os

def reproduce():
    loader = GloSSACLoader()
    
    print("Loading 525 nm data...")
    data_525 = loader.load_raw_data(wavelength_nm=525)
    time, saod_525, saod_file_525 = compute_tropical_saod(data_525)
    
    print("Loading 1020 nm data...")
    data_1020 = loader.load_raw_data(wavelength_nm=1020)
    _, saod_1020, saod_file_1020 = compute_tropical_saod(data_1020)
    
    # Convert time YYYYMM to fractional year for plotting
    years = (time // 100) + (time % 100 - 0.5) / 12.0
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogy(years, saod_525, label='525 nm (GloSSAC Computed)', color='blue', linewidth=1.5)
    plt.semilogy(years, saod_1020, label='1020 nm (GloSSAC Computed)', color='red', linewidth=1.5)
    
    # Styling like Kremser Fig 4
    plt.axhline(0.003, color='gray', linestyle=':', label='Background Ref (~0.003)')
    plt.xlim(1985, years[-1])
    plt.ylim(1e-3, 0.5)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('Tropical Stratospheric SAOD (20°N–20°S)')
    plt.title('GloSSAC V2.23 Tropical Stratospheric SAOD (reproduction)')
    
    # Mark the three background models
    background_targets = {
        'Quiet': 0.00325,
        'Elevated': 0.00998,
        'Post-Eruption': 0.16149
    }
    markers = {'Quiet': 'o', 'Elevated': 's', 'Post-Eruption': 'D'}
    marker_colors = {'Quiet': 'tab:blue', 'Elevated': 'tab:orange', 'Post-Eruption': 'tab:red'}
    
    for name, target in background_targets.items():
        # Find index with SAOD closest to target
        # Prefer points in time ranges that make sense for these conditions
        if name == 'Quiet':
            # Minimum is around 2001
            search_mask = (years > 2000) & (years < 2005)
        elif name == 'Elevated':
            # Nabro/Raikoke period
            search_mask = (years > 2010)
        else: # Post-Eruption
            # Pinatubo period
            search_mask = (years > 1991) & (years < 1995)
            
        masked_saod = np.where(search_mask, saod_525, np.nan)
        idx = np.nanargmin(np.abs(masked_saod - target))
        
        plt.scatter(years[idx], saod_525[idx], color=marker_colors[name], marker=markers[name], 
                    s=80, zorder=5, edgecolors='black', label=f'Model Anchor: {name}')
        plt.annotate(f' {name}', (years[idx], saod_525[idx]), fontsize=9, fontweight='bold',
                     color=marker_colors[name], verticalalignment='bottom')

    plt.legend(fontsize=8, loc='upper right')
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/glossac_tropical_saod.png', dpi=300)
    print("Saved figure to figures/glossac_tropical_saod.png")
    
    # Report epochs
    print("\n--- Key Anchors ---")
    min_idx = np.nanargmin(saod_525)
    print(f"Minimum SAOD (525 nm): {saod_525[min_idx]:.4f} on {time[min_idx]}")
    
    max_idx = np.nanargmax(saod_525)
    print(f"Pinatubo Peak SAOD (525 nm): {saod_525[max_idx]:.4f} on {time[max_idx]}")
    
    # Candidate epochs
    print("\n--- Candidate Epochs for Background Models ---")
    print(f"{'Condition':<15} | {'Date':<8} | {'SAOD_525':<10} | {'SAOD_1020':<10}")
    print("-" * 50)
    
    # Quiet: around May 2001 (minimum)
    quiet_idx = np.nanargmin(np.abs(time - 200105))
    print(f"{'Quiet':<15} | {time[quiet_idx]:<8} | {saod_525[quiet_idx]:.5f} | {saod_1020[quiet_idx]:.5f}")
    
    # Elevated: maybe 2011? (Nabro) or 2019 (Raikoke)
    # Let's find a date with SAOD ~ 0.01
    elevated_idx = np.nanargmin(np.abs(saod_525 - 0.01))
    print(f"{'Elevated':<15} | {time[elevated_idx]:<8} | {saod_525[elevated_idx]:.5f} | {saod_1020[elevated_idx]:.5f}")
    
    # Post-eruption: Pinatubo decay or El Chichon decay
    # Dec 1991 is near peak
    post_idx = np.nanargmin(np.abs(time - 199112))
    print(f"{'Post-Eruption':<15} | {time[post_idx]:<8} | {saod_525[post_idx]:.5f} | {saod_1020[post_idx]:.5f}")

    # Additional candidates
    nabro_idx = np.nanargmin(np.abs(time - 201108))
    print(f"{'Nabro (2011)':<15} | {time[nabro_idx]:<8} | {saod_525[nabro_idx]:.5f} | {saod_1020[nabro_idx]:.5f}")
    
    raikoke_idx = np.nanargmin(np.abs(time - 201910))
    print(f"{'Raikoke (2019)':<15} | {time[raikoke_idx]:<8} | {saod_525[raikoke_idx]:.5f} | {saod_1020[raikoke_idx]:.5f}")

if __name__ == "__main__" and "--paper" not in __import__("sys").argv:
    reproduce()


def paper_figure():
    """Round 41 (Doron, paper1 Fig. glossac_saod): paper-grade version of the
    record with the three states that bound the reservoir's range (pristine
    minimum May 2001, modern volcanically modulated baseline August 2021,
    post-Pinatubo peak December 1991) marked.
    Round 42 (Doron): 525 nm only; the 20-25N bin used by the calibrated
    background (the file's per-bin stratospheric OD, as in
    scripts/glossac_saod_20_25N.py) added for comparison; the Pinatubo
    (June 1991) and Hunga Tonga (January 2022) eruptions marked.
    Writes figures/glossac_tropical_saod.png."""
    loader = GloSSACLoader()
    d = loader.load_raw_data(wavelength_nm=525)
    time, saod_525, _ = compute_tropical_saod(d)
    od = np.ma.masked_invalid(d["od_file"])
    li = int(np.argmin(np.abs(d["lat"] - 22.5)))
    assert abs(d["lat"][li] - 22.5) < 0.1
    od_2025 = od[:, li]
    years = (time // 100) + (time % 100 - 0.5) / 12.0
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
        "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 8.5, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(8.3 / 2.54 * 1.5, 5.5 / 2.54 * 1.5))
    ax.semilogy(years, saod_525, color="black", lw=1.3,   # round 44 (Doron): blue -> black
                label="tropical mean (20$^\\circ$S–20$^\\circ$N)")
    ax.semilogy(years, od_2025, color="tab:red", lw=1.0, alpha=0.9,   # round 44 (Doron): orange -> red
                label="20–25$^\\circ$N bin (calibrated background)")
    marks = [("pristine minimum (May 2001)", 200105, "o"),
             ("modern baseline (Aug 2021)", 202108, "s"),
             ("post-Pinatubo peak (Dec 1991)", 199112, "D")]
    for lab, ym, mk in marks:
        i = int(np.nanargmin(np.abs(time - ym)))
        # round 43 (Doron): hollow symbols so the curves show through
        ax.plot(years[i], saod_525[i], mk, ms=8, color="black", mfc="none", mew=1.1,
                zorder=5, label=lab)
        print(f"  {lab}: {time[i]}  SAOD525 = {saod_525[i]:.4f}  (20-25N {od_2025[i]:.4f})")
    for lab, yr, ha, dx in [("Pinatubo", 1991 + 5.5 / 12, "left", 0.3),
                            ("Hunga Tonga", 2022 + 0.5 / 12, "right", -0.3)]:
        ax.axvline(yr, color="0.35", ls=":", lw=1.0)
        ax.text(yr + dx, 1.15e-3, lab, fontsize=9, color="0.25", ha=ha, va="bottom")
    i_ht = int(np.nanargmax(np.where(years >= 2022.0, saod_525, np.nan)))
    print(f"  post-Hunga Tonga tropical peak: {time[i_ht]}  SAOD525 = {saod_525[i_ht]:.4f}")
    ax.set_xlim(1985, years[-1]); ax.set_ylim(1e-3, 0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Stratospheric AOD at 525 nm")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out = "figures/glossac_tropical_saod.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved paper figure -> {out}")


if __name__ == "__main__" and "--paper" in __import__("sys").argv:
    paper_figure()
