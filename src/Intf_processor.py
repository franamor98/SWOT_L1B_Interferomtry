
"""
SWOT L1B Interferogram Processor

This module processes SWOT L1B HR SLC products to derive:
- interferometric phase
- multilooked coherence
- relative surface height
- approximate geolocation (lat/lon)
- sigma0 backscatter

"""

import numpy as np
import h5py
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


# ==========================================================
# CONSTANTS
# ==========================================================

# Reference Earth radius [m], spherical approximation
R_EARTH = 6378137.0


# ==========================================================
# BASIC HELPERS
# ==========================================================

def to_complex(arr):
    """
    Convert SWOT SLC arrays to complex format.

    Parameters
    ----------
    arr : ndarray
        Input array, either complex-valued or with last dimension
        containing [real, imaginary].

    Returns
    -------
    ndarray
        Complex-valued array.
    """
    a = arr[...]
    if np.iscomplexobj(a):
        return a
    if a.shape[-1] == 2:
        return a[..., 0].astype(np.float32) + 1j * a[..., 1].astype(np.float32)
    raise ValueError("Input is neither complex nor {real,imag} format.")


def make_h0_s_full(f, Ls_full, Ps_full):
    """
    Resample GRDEM reference height to the SLC grid.

    Handles correct selection of left/right swath and aligns
    the inner swath edge with the SLC grid.

    Parameters
    ----------
    f : h5py.File
        Open SWOT L1B file handle.
    Ls_full : int
        Number of azimuth lines in SLC.
    Ps_full : int
        Number of range pixels in SLC.

    Returns
    -------
    ndarray
        Resampled reference surface height on the SLC grid.
    """
    h_grdem = f["grdem/height"][:]
    Lh, Ph = h_grdem.shape

    side = f.attrs.get("swath_side", b"R").decode()
    j0 = Ph // 2

    # Select correct swath half and orientation
    if side == "R":
        blk = h_grdem[:, j0:]
    else:
        blk = h_grdem[:, :j0][:, ::-1]

    # Interpolate onto SLC grid
    y_src = np.linspace(0, blk.shape[0] - 1, blk.shape[0])
    x_src = np.linspace(0, blk.shape[1] - 1, blk.shape[1])
    interp = RegularGridInterpolator((y_src, x_src), blk, fill_value=np.nan)

    y_dst = np.linspace(0, blk.shape[0] - 1, Ls_full)
    x_dst = np.linspace(0, blk.shape[1] - 1, Ps_full)
    Y, X = np.meshgrid(y_dst, x_dst, indexing="ij")

    return interp(np.column_stack([Y.ravel(), X.ravel()])).reshape(Ls_full, Ps_full)


# ==========================================================
# MULTILOOKING
# ==========================================================

def goldstein_filter_block(I, alpha=0.4):
    """
    Apply Goldstein phase filter to a complex block.

    Parameters
    ----------
    I : ndarray
        Complex interferogram block.
    alpha : float
        Goldstein filter exponent.

    Returns
    -------
    ndarray
        Filtered complex block.
    """
    F = np.fft.fft2(I)
    H = np.abs(F) ** alpha
    return np.fft.ifft2(F * H)


def multilook_goldstein(I, alpha=0.4, block=(64, 64)):
    """
    Apply block-wise Goldstein filtering to an interferogram.

    Parameters
    ----------
    I : ndarray
        Complex interferogram.
    alpha : float
        Goldstein exponent.
    block : tuple
        Block size (azimuth, range).

    Returns
    -------
    ndarray
        Goldstein-filtered interferogram.
    """
    I_gold = np.zeros_like(I, dtype=np.complex64)
    by, bx = block

    for y in range(0, I.shape[0], by):
        for x in range(0, I.shape[1], bx):
            patch = I[y:y + by, x:x + bx]
            if patch.size:
                I_gold[y:y + by, x:x + bx] = goldstein_filter_block(patch, alpha)

    return I_gold


def multilook_gamma(S1, S2, window=(32, 4)):
    """
    Coherence-weighted (gamma-based) multilooking.

    Parameters
    ----------
    S1, S2 : ndarray
        Complex SLC channels.
    window : tuple
        Multilook window size (azimuth, range).

    Returns
    -------
    I_ml : ndarray
        Multilooked interferogram.
    P1_ml, P2_ml : ndarray
        Multilooked powers.
    gamma_ml : ndarray
        Estimated coherence.
    """
    I = S1 * np.conj(S2)
    P1 = np.abs(S1) ** 2
    P2 = np.abs(S2) ** 2

    # Initial multilook
    I0 = uniform_filter(I.real, size=window) + 1j * uniform_filter(I.imag, size=window)
    P1_0 = uniform_filter(P1, size=window)
    P2_0 = uniform_filter(P2, size=window)

    gamma0 = np.abs(I0) / (np.sqrt(P1_0 * P2_0) + 1e-12)

    # Gamma-squared weighting
    w = gamma0 ** 2
    W = uniform_filter(w, size=window) + 1e-12

    I_ml = (uniform_filter(I.real * w, size=window) +
            1j * uniform_filter(I.imag * w, size=window)) / W

    P1_ml = uniform_filter(P1 * w, size=window) / W
    P2_ml = uniform_filter(P2 * w, size=window) / W

    gamma_ml = np.abs(I_ml) / (np.sqrt(P1_ml * P2_ml) + 1e-12)

    return I_ml, P1_ml, P2_ml, gamma_ml


# ==========================================================
# GEOLOCATION
# ==========================================================

def ecef_to_llh(x, y, z):
    """
    Convert ECEF coordinates to latitude, longitude, and height.

    Parameters
    ----------
    x, y, z : ndarray
        ECEF coordinates [m].

    Returns
    -------
    lat : ndarray
        Geodetic latitude [rad].
    lon : ndarray
        Longitude [rad].
    h : ndarray
        Height above ellipsoid [m].
    """
    a = 6378137.0
    e2 = 6.69437999014e-3

    lon = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, r * (1 - e2))

    # Iterative latitude refinement
    for _ in range(2):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat ** 2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - e2 * N / (N + h)))

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    h = r / np.cos(lat) - N
    return lat, lon, h  # height not used downstream


def geolocate_swot(h_abs, psi, sat_pos, sat_vel, side):
    """
    Approximate geolocation of SWOT pixels.

    Uses spherical Earth geometry and cross-track rotation from
    nadir defined by psi.

    Parameters
    ----------
    h_abs : ndarray
        Absolute surface height [m].
    psi : ndarray
        Central angle between nadir and pixel.
    sat_pos : ndarray
        Satellite position vectors [m].
    sat_vel : ndarray
        Satellite velocity vectors [m/s].
    side : str
        Swath side ('L' or 'R').

    Returns
    -------
    lat, lon : ndarray
        Geodetic latitude and longitude [deg].
    h : ndarray
        Ellipsoidal height (approximate).
    """
    r_hat = sat_pos / np.linalg.norm(sat_pos, axis=1, keepdims=True)
    v_hat = sat_vel / np.linalg.norm(sat_vel, axis=1, keepdims=True)

    # Cross-track direction
    c_hat = np.cross(v_hat, r_hat) 
    c_hat /= np.linalg.norm(c_hat, axis=1, keepdims=True)

    if side.upper() == "L":
        c_hat = -c_hat

    # Broadcast over range
    r_hat = r_hat[:, None, :]
    c_hat = c_hat[:, None, :]

    cos_psi = np.cos(psi)[..., None]
    sin_psi = np.sin(psi)[..., None]

    # Direction from Earth center to ground pixel
    g_hat = cos_psi * r_hat + sin_psi * c_hat

    radius = (R_EARTH + h_abs)[..., None]
    X = radius * g_hat

    lat, lon, h = ecef_to_llh(X[..., 0], X[..., 1], X[..., 2])
    return np.degrees(lat), np.degrees(lon), h


# ==========================================================
# MAIN PROCESSOR CLASS
# ==========================================================
class SWOTL1InterferogramProcessor:
    """
    SWOT L1B interferogram processing pipeline.

    This class reads SWOT L1B HR SLC data and derives:
    - interferometric phase
    - multilooked coherence
    - relative surface height
    - approximate geolocation (lat/lon)
    - sigma0 backscatter
    """

    def __init__(
        self,
        filename,
        window=(32, 4),
        swath_side="R",
        golstein=False,
        alpha=0.4,
        block=(128, 128),
        gamma_thresh=0.3,
        cut_nadir_index=200,
    ):
        """
        Initialize processor configuration.

        Parameters
        ----------
        filename : str
            Path to SWOT L1B file.
        window : tuple
            Multilooking window (azimuth, range).
        swath_side : str
            Swath side ('L' or 'R').
        golstein : bool
            Apply Goldstein filtering after multilooking.
        alpha : float
            Goldstein filter exponent.
        block : tuple
            Goldstein filter block size.
        gamma_thresh : float
            Coherence threshold for height filtering.
        cut_nadir_index : int
            Pixels to skip near nadir.
        """
        self.filename = filename
        self.window = window
        self.swath_side = swath_side
        self.golstein = golstein
        self.alpha = alpha
        self.block = block
        self.gamma_thresh = gamma_thresh
        self.cut_nadir_index = cut_nadir_index

    # --------------------------------------------------
    def run(self):
        """
        Run the full processing chain in sequence.
        """
        self._read_l1()
        self._geometry()
        self._interferogram()
        self._multilook()
        self._height()
        self._geolocate()
        self._sigma0()
        self._filter_height()

    # --------------------------------------------------
    def _read_l1(self):
        """
        Read L1B input data and prepare core geometry variables.

        - Reads SLC channels
        - Resamples reference DEM
        - Identifies nadir region and trims swath
        - Extracts satellite position, velocity, altitude, and antenna positions
        """
        with h5py.File(self.filename, "r") as f:
            ga = f.attrs

            # Radar parameters
            self.dr = ga["nominal_slant_range_spacing"][0]
            self.R0 = ga["near_range"][0]
            self.lam = ga["wavelength"][0]

            # Read complex SLC channels
            self.S1_full = to_complex(f["slc/slc_plus_y"][:])
            self.S2_full = to_complex(f["slc/slc_minus_y"][:])
            self.Ls_full, self.Ps_full = self.S1_full.shape

            # Reference surface height on SLC grid
            self.h0_s_full = make_h0_s_full(f, self.Ls_full, self.Ps_full)

            # Estimate nadir column from mean interferometric phase
            I0 = self.S1_full * np.conj(self.S2_full)
            phi_col = np.angle(np.mean(I0, axis=0))
            self.icut = (
                uniform_filter(np.unwrap(phi_col), 50).argmax()
                + self.cut_nadir_index
            )

            # Trim data away from nadir
            self.S1 = self.S1_full[:, self.icut:]
            self.S2 = self.S2_full[:, self.icut:]
            self.h0_s = self.h0_s_full[:, self.icut:]
            self.Ls, self.Ps = self.S1.shape

            # Slant range for each pixel
            rng = self.R0 + self.dr * np.arange(self.icut, self.icut + self.Ps)
            self.R = np.broadcast_to(rng[None, :], (self.Ls, self.Ps))

            # Satellite position, velocity, and altitude
            tvp = f["tvp"]
            pos = np.c_[tvp["x"][:], tvp["y"][:], tvp["z"][:]]
            vel = np.c_[tvp["vx"][:], tvp["vy"][:], tvp["vz"][:]]

            i0 = ga["slc_first_line_index_in_tvp"][0]
            i1 = ga["slc_last_line_index_in_tvp"][0]

            self.sat_pos = pos[i0:i1 + 1]
            self.sat_vel = vel[i0:i1 + 1]

            # Interpolate altitude to SLC azimuth grid
            alt = tvp["altitude"][i0:i1 + 1]
            self.alt = np.interp(
                np.linspace(0, 1, self.Ls),
                np.linspace(0, 1, len(alt)),
                alt,
            )

            # Antenna positions for baseline estimation
            self.Aplus = np.c_[
                tvp["plus_y_antenna_x"][:],
                tvp["plus_y_antenna_y"][:],
                tvp["plus_y_antenna_z"][:],
            ]
            self.Aminus = np.c_[
                tvp["minus_y_antenna_x"][:],
                tvp["minus_y_antenna_y"][:],
                tvp["minus_y_antenna_z"][:],
            ]

    # --------------------------------------------------
    def _geometry(self):
        """
        Compute geometric quantities required for height inversion.

        - Central angle (psi)
        - Look and incidence geometry
        - Perpendicular baseline
        - Height sensitivity factor S
        """
        # Radii of satellite and reference surface
        rp = R_EARTH + self.alt[:, None]
        rs = R_EARTH + self.h0_s

        # Central angle between nadir and pixel
        cos_psi = (rp**2 + rs**2 - self.R**2) / (2 * rp * rs)
        cos_psi = np.clip(cos_psi, -1, 1)
        self.psi = np.arccos(cos_psi)

        # Look angle geometry
        sin_psi = np.sqrt(1 - cos_psi**2)
        sin_th_look = (rs / self.R) * sin_psi

        # Incidence angle cosine
        cos_th_inc = (rp**2 - rs**2 - self.R**2) / (2 * rs * self.R)
        cos_th_inc = np.clip(cos_th_inc, -1, 1)

        # Approximate perpendicular baseline magnitude
        B = np.linalg.norm(self.Aminus.mean(0) - self.Aplus.mean(0))
        Bperp = B * cos_th_inc
        Bperp[Bperp < 1.0] = np.nan

        # Height-phase sensitivity factor
        self.S = (
            self.lam * self.R * sin_th_look
            / (2 * np.pi * (Bperp + 1e-9))
        )

    # --------------------------------------------------
    def _interferogram(self):
        """
        Form interferogram and remove long-wavelength phase ramps.
        """
        I0 = self.S1 * np.conj(self.S2)

        # Mean phase across azimuth for flattening
        phi = np.angle(np.mean(I0, axis=0))
        phi_smooth = uniform_filter(np.unwrap(phi), 50)

        # Polynomial phase ramp removal
        x = np.arange(self.Ps)
        phi_fit = np.polyval(np.polyfit(x, phi_smooth, 15), x)[None, :]

        self.S1_d = self.S1 * np.exp(-1j * phi_fit / 2)
        self.S2_d = self.S2 * np.exp(+1j * phi_fit / 2)

    # --------------------------------------------------
    def _multilook(self):
        """
        Multilook the interferogram and estimate coherence.
        """
        I_ml, self.P1_ml, self.P2_ml, self.gamma_ml = multilook_gamma(
            self.S1_d, self.S2_d, self.window
        )

        # Optional Goldstein filtering
        if self.golstein:
            I_ml = multilook_goldstein(
                I_ml, alpha=self.alpha, block=self.block
            )

        # Interferometric phase
        self.phi_ml = -np.angle(I_ml)

    # --------------------------------------------------
    def _height(self):
        """
        Convert interferometric phase to absolute height.
        """
        self.h_abs = self.h0_s + self.S * self.phi_ml

    # --------------------------------------------------
    def _geolocate(self):
        """
        Compute approximate latitude and longitude of pixels.
        """
        self.lat, self.lon, _ = geolocate_swot(
            self.h_abs,
            self.psi,
            self.sat_pos,
            self.sat_vel,
            self.swath_side,
        )

    # --------------------------------------------------
    def _sigma0(self):
        """
        Compute sigma0 backscatter using x-factor calibration.
        """
        with h5py.File(self.filename, "r") as f:
            xf_p = f["xfactor/xfactor_plus_y"][:, self.icut:]
            xf_m = f["xfactor/xfactor_minus_y"][:, self.icut:]

        s0 = 0.5 * (self.P1_ml / xf_p + self.P2_ml / xf_m)
        self.sigma0_db = 10 * np.log10(s0 + 1e-12)

    # --------------------------------------------------
    def _filter_height(self):
        """
        Filter height estimates using coherence and percentile thresholds.
        """
        h = self.h_abs.copy()

        # Mask low-coherence pixels
        h[self.gamma_ml < self.gamma_thresh] = np.nan

        # Remove extreme outliers
        p1, p99 = np.nanpercentile(h, [1, 99])
        h[(h < p1) | (h > p99)] = np.nan

        self.h_abs_filtered = h

    # --------------------------------------------------
    def to_xarray(self):
        """
        Export outputs as an xarray.Dataset.
        """
        ds = xr.Dataset(
            data_vars=dict(
                height=(("line", "pixel"), self.h_abs),
                height_filtered=(("line", "pixel"), self.h_abs_filtered),
                sigma0=(("line", "pixel"), self.sigma0_db),
                coherence=(("line", "pixel"), self.gamma_ml),
                phase=(("line", "pixel"), self.phi_ml),
            ),
            coords=dict(
                latitude=(("line", "pixel"), self.lat),
                longitude=(("line", "pixel"), self.lon),
            ),
            attrs=dict(
                title="SWOT L1 interferometric derived fields",
                source="SWOTL1InterferogramProcessor",
                conventions="CF-1.7",
            ),
        )
        return ds
