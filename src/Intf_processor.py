import numpy as np
import h5py
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

# ==========================================================
# CONSTANTS
# ==========================================================
R_EARTH = 6378137.0


# ==========================================================
# BASIC HELPERS
# ==========================================================
def to_complex(arr):
    """Convert SWOT real/imag array to complex."""
    a = arr[...]
    if np.iscomplexobj(a):
        return a
    if a.shape[-1] == 2:
        return a[..., 0].astype(np.float32) + 1j * a[..., 1].astype(np.float32)
    raise ValueError("Input is neither complex nor {real,imag} format.")


def make_h0_s_full(f, Ls_full, Ps_full):
    """Resample GRDEM height to SLC grid (correct wing, inner-edge aligned)."""
    h_grdem = f["grdem/height"][:]
    Lh, Ph = h_grdem.shape

    side = f.attrs.get("swath_side", b"R").decode()
    j0 = Ph // 2

    if side == "R":
        blk = h_grdem[:, j0:]
    else:
        blk = h_grdem[:, :j0][:, ::-1]

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
    F = np.fft.fft2(I)
    H = np.abs(F) ** alpha
    return np.fft.ifft2(F * H)


def multilook_goldstein(I, alpha=0.4, block=(64, 64)):
    I_gold = np.zeros_like(I, dtype=np.complex64)
    by, bx = block

    for y in range(0, I.shape[0], by):
        for x in range(0, I.shape[1], bx):
            patch = I[y:y + by, x:x + bx]
            if patch.size:
                I_gold[y:y + by, x:x + bx] = goldstein_filter_block(patch, alpha)

    return I_gold


def multilook_gamma(S1, S2, window=(32, 4)):
    """Gamma-weighted multilooking."""
    I = S1 * np.conj(S2)
    P1 = np.abs(S1) ** 2
    P2 = np.abs(S2) ** 2

    I0 = uniform_filter(I.real, size=window) + 1j * uniform_filter(I.imag, size=window)
    P1_0 = uniform_filter(P1, size=window)
    P2_0 = uniform_filter(P2, size=window)

    gamma0 = np.abs(I0) / (np.sqrt(P1_0 * P2_0) + 1e-12)
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
    """Convert ECEF arrays to WGS84 lat/lon/h."""
    a = 6378137.0
    e2 = 6.69437999014e-3

    lon = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, r * (1 - e2))

    for _ in range(2):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat ** 2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - e2 * N / (N + h)))

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    h = r / np.cos(lat) - N
    return lat, lon, h


def geolocate_swot(h_abs, psi, sat_pos, sat_vel, side):
    """Geolocate SWOT pixels from height + psi + satellite vectors."""
    r_hat = sat_pos / np.linalg.norm(sat_pos, axis=1, keepdims=True)
    v_hat = sat_vel / np.linalg.norm(sat_vel, axis=1, keepdims=True)
    c_hat = np.cross(v_hat, r_hat)
    c_hat /= np.linalg.norm(c_hat, axis=1, keepdims=True)

    if side.upper() == "L":
        c_hat = -c_hat

    r_hat = r_hat[:, None, :]
    c_hat = c_hat[:, None, :]

    cos_psi = np.cos(psi)[..., None]
    sin_psi = np.sin(psi)[..., None]

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
    SWOT L1B interferogram → multilooked height → geolocation processor.
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
        with h5py.File(self.filename, "r") as f:
            ga = f.attrs
            self.dr = ga["nominal_slant_range_spacing"][0]
            self.R0 = ga["near_range"][0]
            self.lam = ga["wavelength"][0]

            self.S1_full = to_complex(f["slc/slc_plus_y"][:])
            self.S2_full = to_complex(f["slc/slc_minus_y"][:])
            self.Ls_full, self.Ps_full = self.S1_full.shape

            self.h0_s_full = make_h0_s_full(f, self.Ls_full, self.Ps_full)

            I0 = self.S1_full * np.conj(self.S2_full)
            phi_col = np.angle(np.mean(I0, axis=0))
            self.icut = (
                uniform_filter(np.unwrap(phi_col), 50).argmax()
                + self.cut_nadir_index
            )

            self.S1 = self.S1_full[:, self.icut:]
            self.S2 = self.S2_full[:, self.icut:]
            self.h0_s = self.h0_s_full[:, self.icut:]
            self.Ls, self.Ps = self.S1.shape

            rng = self.R0 + self.dr * np.arange(self.icut, self.icut + self.Ps)
            self.R = np.broadcast_to(rng[None, :], (self.Ls, self.Ps))

            tvp = f["tvp"]
            pos = np.c_[tvp["x"][:], tvp["y"][:], tvp["z"][:]]
            vel = np.c_[tvp["vx"][:], tvp["vy"][:], tvp["vz"][:]]

            i0 = ga["slc_first_line_index_in_tvp"][0]
            i1 = ga["slc_last_line_index_in_tvp"][0]

            self.sat_pos = pos[i0:i1 + 1]
            self.sat_vel = vel[i0:i1 + 1]

            alt = tvp["altitude"][i0:i1 + 1]
            self.alt = np.interp(
                np.linspace(0, 1, self.Ls),
                np.linspace(0, 1, len(alt)),
                alt,
            )

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
        rp = R_EARTH + self.alt[:, None]
        rs = R_EARTH + self.h0_s

        cos_psi = (rp**2 + rs**2 - self.R**2) / (2 * rp * rs)
        cos_psi = np.clip(cos_psi, -1, 1)
        self.psi = np.arccos(cos_psi)

        sin_psi = np.sqrt(1 - cos_psi**2)
        sin_th_look = (rs / self.R) * sin_psi

        cos_th_inc = (rp**2 - rs**2 - self.R**2) / (2 * rs * self.R)
        cos_th_inc = np.clip(cos_th_inc, -1, 1)

        B = np.linalg.norm(self.Aminus.mean(0) - self.Aplus.mean(0))
        Bperp = B * cos_th_inc
        Bperp[Bperp < 1.0] = np.nan

        self.S = (
            self.lam * self.R * sin_th_look
            / (2 * np.pi * (Bperp + 1e-9))
        )

    # --------------------------------------------------
    def _interferogram(self):
        I0 = self.S1 * np.conj(self.S2)
        phi = np.angle(np.mean(I0, axis=0))
        phi_smooth = uniform_filter(np.unwrap(phi), 50)

        x = np.arange(self.Ps)
        phi_fit = np.polyval(np.polyfit(x, phi_smooth, 15), x)[None, :]

        self.S1_d = self.S1 * np.exp(-1j * phi_fit / 2)
        self.S2_d = self.S2 * np.exp(+1j * phi_fit / 2)

    # --------------------------------------------------
    def _multilook(self):
        I_ml, self.P1_ml, self.P2_ml, self.gamma_ml = multilook_gamma(
            self.S1_d, self.S2_d, self.window
        )

        if self.golstein:
            I_ml = multilook_goldstein(
                I_ml, alpha=self.alpha, block=self.block
            )

        self.phi_ml = -np.angle(I_ml)

    # --------------------------------------------------
    def _height(self):
        self.h_abs = self.h0_s + self.S * self.phi_ml

    # --------------------------------------------------
    def _geolocate(self):
        self.lat, self.lon, _ = geolocate_swot(
            self.h_abs,
            self.psi,
            self.sat_pos,
            self.sat_vel,
            self.swath_side,
        )

    # --------------------------------------------------
    def _sigma0(self):
        with h5py.File(self.filename, "r") as f:
            xf_p = f["xfactor/xfactor_plus_y"][:, self.icut:]
            xf_m = f["xfactor/xfactor_minus_y"][:, self.icut:]

        s0 = 0.5 * (self.P1_ml / xf_p + self.P2_ml / xf_m)
        self.sigma0_db = 10 * np.log10(s0 + 1e-12)

    # --------------------------------------------------
    def _filter_height(self):
        h = self.h_abs.copy()
        h[self.gamma_ml < self.gamma_thresh] = np.nan

        p1, p99 = np.nanpercentile(h, [1, 99])
        h[(h < p1) | (h > p99)] = np.nan

        self.h_abs_filtered = h

    # --------------------------------------------------
    def to_xarray(self):
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
