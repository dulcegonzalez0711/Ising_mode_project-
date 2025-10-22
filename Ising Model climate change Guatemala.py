from __future__ import annotations
import os, math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from typing import Optional, Dict

try:
    import pandas as pd
except Exception:
    pd = None

SPIN_CMAP = ListedColormap(["blue", "red"]) 
FAST = True

J = 1.0
KB = 1.0

TEMPERATURE = 2.5

H_FIELD = 0.0

if FAST:
    L = 32
    SWEEPS_EQ = 1500
    BURN_IN = 300
    SEEDS_PER_T = 1
    T_MIN, T_MAX, T_STEP = 1.5, 3.2, 0.05
    SWEEPS_DYN = 1500
    V6_STEPS = 12
else:
    L = 64
    SWEEPS_EQ = 3000
    BURN_IN = 1500
    SEEDS_PER_T = 3
    T_MIN, T_MAX, T_STEP = 1.5, 3.5, 0.2
    SWEEPS_DYN = 1500
    V6_STEPS = 24

# Configuraciones iniciales
BALANCED_INIT = True                 # 50/50 aleatorio cuando es None
REALISTIC_INIT = True                # Correr también con sesgo “realista”
REALISTIC_FRACTION_WARM = 0.60       # 60% cálido (+1) para el caso realista

USE_BERKELEY_FIELD = False
BERKELEY_PATH = r"c:\Users\Deltaz\Downloads\guatemala-TAVG-Trend.csv"
ALPHA_H = 0.5 

def random_spins(L: int, frac_plus: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crea una malla LxL de espines con fracción +1 controlada (o 50/50 si None)."""
    rng = rng or np.random.default_rng()
    if frac_plus is None:
        return rng.choice([-1, 1], size=(L, L))
    n = L * L
    k = int(round(frac_plus * n))
    arr = np.array([1]*k + [-1]*(n-k), dtype=int)
    rng.shuffle(arr)
    return arr.reshape(L, L)

def metropolis_sweeps(
    spins: np.ndarray,
    T: float,
    sweeps: int = 500,
    burn_in: int = 500,
    rng: np.random.Generator | None = None
) -> Dict[str, np.ndarray]:
    """Barridos de Metrópolis con almacenamiento de series de |M| y E después de burn-in."""
    rng = rng or np.random.default_rng()
    Lx = spins.shape[0]
    n = Lx * Lx
    h = H_FIELD

    def dE(i: int, j: int) -> float:
        nn = (spins[(i+1) % Lx, j] + spins[(i-1) % Lx, j] +
              spins[i, (j+1) % Lx] + spins[i, (j-1) % Lx])
        return 2.0 * spins[i, j] * (J * nn + h)

    def energy_total() -> float:
        E = 0.0
        E -= J * np.sum(spins * np.roll(spins, -1, axis=0))
        E -= J * np.sum(spins * np.roll(spins, -1, axis=1))
        E -= h * np.sum(spins)
        return float(E)

    Ms, Es = [], []
    E_cur = energy_total()
    for s in range(sweeps):
        I = rng.integers(0, Lx, size=n)
        Jidx = rng.integers(0, Lx, size=n)
        for i, j in zip(I, Jidx):
            de = dE(i, j)
            if de <= 0.0 or rng.random() < math.exp(-de / (KB*T)):
                spins[i, j] *= -1
                E_cur += de
        M_cur = np.sum(spins)
        if s >= burn_in:
            Ms.append(abs(M_cur) / n)    # |M| por sitio
            Es.append(E_cur / n)         # E por sitio
    return {"M": np.array(Ms), "E": np.array(Es), "spins": spins}

def observables_from_series(M: np.ndarray, E: np.ndarray, T: float) -> Dict[str, float]:
    """Medias y fluctuaciones -> χ y C."""
    M_mean, M2_mean = float(np.mean(M)), float(np.mean(M**2))
    E_mean, E2_mean = float(np.mean(E)), float(np.mean(E**2))
    chi = (M2_mean - M_mean**2) / (KB*T)
    C = (E2_mean - E_mean**2) / (KB*T**2)
    return {"M": M_mean, "chi": chi, "C": C, "E": E_mean}

def _read_berkeley_table(path: str) -> Optional["pd.DataFrame"]:
    if (pd is None) or (not os.path.exists(path)):
        return None
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    lines = [ln.strip().strip('"') for ln in lines]
    import re
    first_idx = None
    for idx, ln in enumerate(lines):
        if re.match(r"^\s*\d{4}\s", ln):
            first_idx = idx; break
    if first_idx is None:
        return None
    rows = []
    for ln in lines[first_idx:]:
        if not ln or ln.startswith("%"): 
            continue
        parts = ln.split()
        parts = parts[:12] + ["NaN"]*(12-len(parts)) if len(parts) < 12 else parts[:12]
        rows.append(parts)
    cols = ["Year","Month","Mon_Anom","Mon_Unc","Ann_Anom","Ann_Unc",
            "FiveY_Anom","FiveY_Unc","TenY_Anom","TenY_Unc","TwentyY_Anom","TwentyY_Unc"]
    df = pd.DataFrame(rows, columns=cols).apply(pd.to_numeric, errors="coerce")
    return df

def h_from_berkeley(path: str) -> float:
    """Convierte la anomalía reciente (promedio móvil 12m) en un h efectivo."""
    if pd is None:
        raise RuntimeError("pandas requerido")
    df = _read_berkeley_table(path)
    if df is None:
        raise RuntimeError("No se pudo leer el archivo de Berkeley Earth")
    df["roll12"] = df["Mon_Anom"].rolling(12, min_periods=6).mean()
    recent = df["roll12"].dropna().iloc[-1] if df["roll12"].notna().any() else df["Mon_Anom"].dropna().iloc[-1]
    return ALPHA_H * float(recent)

def climate_series_from_berkeley(path: str) -> Optional[dict]:
    if (pd is None) or (not os.path.exists(path)):
        return None
    df = _read_berkeley_table(path)
    if df is None:
        return None

    df = df.dropna(subset=["Mon_Anom", "Mon_Unc"])
    group = df.groupby("Year", as_index=False)[["Mon_Anom","Mon_Unc"]].mean()
    if group.empty:
        return None
    years = group["Year"].to_numpy()
    anom  = group["Mon_Anom"].to_numpy()
    unc   = group["Mon_Unc"].to_numpy()
    x = years - years[0]
    coeff = np.polyfit(x, anom, 1)
    slope_per_year = coeff[0]
    trend_century = slope_per_year * 100.0
    return {"years": years, "anom": anom, "unc": unc, "trend": trend_century}

def recent_anomaly_from_berkeley(path: str) -> float:
    if (pd is None) or (not os.path.exists(path)):
        return 0.6
    df = _read_berkeley_table(path)
    if df is None:
        return 0.6
    df = df.dropna(subset=["Mon_Anom"])
    df["roll12"] = df["Mon_Anom"].rolling(12, min_periods=6).mean()
    if df["roll12"].notna().any():
        return float(df["roll12"].dropna().iloc[-1])
    return float(df["Mon_Anom"].dropna().iloc[-1])

def soft_map_anomaly_to_frac_plus(anomaly_celsius: float) -> float:
    """Mapeo suave de anomalía térmica (°C) a fracción de +1 al inicio."""
    M0 = 0.5 + float(anomaly_celsius) / 4.0
    frac_plus = 0.5 * (1.0 + M0)
    return float(np.clip(frac_plus, 0.05, 0.95))

def plot_initial(spins_init: np.ndarray, frac_plus: float | None, clima_series=None, title_suffix=""):
    fig = plt.figure(figsize=(12, 6.5))
    fig.suptitle(f"Datos Climáticos y Configuración del Sistema {title_suffix}".strip(),
                 fontsize=16, weight="bold")

    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    if clima_series is not None:
        years = clima_series["years"]
        anom  = clima_series["anom"]
        unc   = clima_series.get("unc", None)
        trend = clima_series.get("trend", None)
        ax1.plot(years, anom, label="Anomalía térmica")
        if unc is not None:
            ax1.fill_between(years, anom-unc, anom+unc, alpha=0.25, label="Incertidumbre")
        if trend is not None:
            x = np.array(years)
            yfit = (x - x[0]) * (trend/100.0) + anom[0]
            ax1.plot(years, yfit, linestyle="--", label=f"Tendencia: {trend:.2f}°C/siglo")
        ax1.set_title("Datos Climáticos (Entrada del Sistema)")
        ax1.set_xlabel("Año")
        ax1.set_ylabel("Anomalía (°C)")
        ax1.legend(loc="best")
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "Sin datos climáticos\n(opcional)", ha="center", va="center", fontsize=12)

    ax2 = plt.subplot2grid((2, 3), (0, 2))
    im = ax2.imshow(spins_init, interpolation="nearest", vmin=-1, vmax=1, cmap=SPIN_CMAP)
    ax2.set_title("Configuración de Espines\n$s_i=+1$ (cálido), $-1$ (frío)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_ticks([-1, 1]); cbar.set_ticklabels(["Frío (-1)", "Cálido (+1)"])

    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    vals = spins_init.ravel()
    ax3.bar(["-1 (Frío)", "+1 (Cálido)"], [int(np.sum(vals==-1)), int(np.sum(vals==1))], color=["blue","red"])
    ax3.set_title("Distribución Inicial de Espines")
    ax3.set_xlabel("Espín"); ax3.set_ylabel("Recuento")
    if frac_plus is not None:
        ax3.text(0.98, 0.92, f"{frac_plus*100:.1f}% cálido", ha="right", va="center", transform=ax3.transAxes)

    plt.tight_layout(); plt.show()

def plot_thermo(Ts, Ms, chis, Cs, Es, Tc_guess=None, title_suffix=""):
    plt.figure(figsize=(12, 6.5))
    plt.suptitle(f"Transición de Fase y Propiedades Termodinámicas {title_suffix}".strip(),
                 fontsize=16, weight="bold")

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(Ts, Ms, marker="o", label="|M|")
    if Tc_guess: ax1.axvline(Tc_guess, linestyle="--", label=f"Tc = {Tc_guess:.3f}")
    ax1.set_title("Magnetización")
    ax1.set_xlabel("Temperatura (J/k_B)"); ax1.set_ylabel("|M|")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(Ts, chis, marker="o", label="χ")
    if Tc_guess: ax2.axvline(Tc_guess, linestyle="--", label=f"Tc = {Tc_guess:.3f}")
    ax2.set_title("Susceptibilidad")
    ax2.set_xlabel("Temperatura (J/k_B)"); ax2.set_ylabel("χ")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(Ts, Cs, marker="o", label="C")
    if Tc_guess: ax3.axvline(Tc_guess, linestyle="--", label=f"Tc = {Tc_guess:.3f}")
    ax3.set_title("Capacidad Calorífica")
    ax3.set_xlabel("Temperatura (J/k_B)"); ax3.set_ylabel("C")
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(Ts, Es, marker="s", label="E")
    if Tc_guess: ax4.axvline(Tc_guess, linestyle="--", label=f"Tc = {Tc_guess:.3f}")
    ax4.set_title("Energía Interna")
    ax4.set_xlabel("Temperatura (J/k_B)"); ax4.set_ylabel("E")
    ax4.legend()

    plt.tight_layout(); plt.show()

def plot_dynamics_and_critical(T: float, M_t, E_t, M_eq_samples, Ts_zoom, Ms_zoom, Tc_guess=None, title_suffix=""):
    plt.figure(figsize=(12, 6.5))
    plt.suptitle(f"Dinámica y Comportamiento Crítico {title_suffix}".strip(),
                 fontsize=16, weight="bold")

    # Serie temporal de convergencia
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(M_t, label="|M|")
    e_norm = (E_t - np.min(E_t)) / max(1e-12, (np.max(E_t)-np.min(E_t)))
    ax1.plot(e_norm, label="E (normalizada)")
    ax1.set_title(f"Convergencia (T = {T:.2f})")
    ax1.set_xlabel("Pasos de Monte Carlo"); ax1.set_ylabel("Magnetización / Energía")
    ax1.legend()

    # Histograma de |M| (equilibrio)
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(M_eq_samples, bins=20, alpha=0.85)
    mu, sigma = float(np.mean(M_eq_samples)), float(np.std(M_eq_samples))
    ax2.axvline(mu, linestyle="--", label=f"⟨|M|⟩ = {mu:.3f}")
    ax2.axvline(mu+sigma, color="gold", linestyle=":", label=f"σ = {sigma:.3f}")
    ax2.axvline(mu-sigma, color="gold", linestyle=":")
    ax2.set_title("Histograma de |M| (equilibrio)")
    ax2.set_xlabel("|M|"); ax2.set_ylabel("Densidad")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(Ts_zoom, Ms_zoom, marker="o", label="|M|")
    if Tc_guess: ax3.axvline(Tc_guess, linestyle="--", label=f"Tc = {Tc_guess:.3f}")
    ax3.set_title("Zoom crítico")
    ax3.set_xlabel("Temperatura (J/k_B)"); ax3.set_ylabel("|M|")
    ax3.legend()

    # Diagrama de fases
    ax4 = plt.subplot(2, 2, 4)
    T_dense = np.linspace(min(Ts_zoom), max(Ts_zoom), 200)
    Ms_interp = np.interp(T_dense, Ts_zoom, Ms_zoom)
    ax4.plot(T_dense, Ms_interp, label="|M|(T)")
    if Tc_guess:
        ax4.axvline(Tc_guess, linestyle="--", label=f"Tc = {Tc_guess:.3f}")
    ax4.fill_between(T_dense, Ms_interp, 0, where=T_dense <= (Tc_guess or 2.27), alpha=0.15)
    ax4.fill_between(T_dense, 0, 0.05, where=T_dense > (Tc_guess or 2.27), alpha=0.10,
                     label="Paramagnética (|M|≈0)")
    ax4.set_ylim(bottom=0)
    ax4.set_title("Diagrama de fases (esquemático)")
    ax4.set_xlabel("Temperatura (J/k_B)"); ax4.set_ylabel("|M|")
    ax4.legend()

    plt.tight_layout(); plt.show()

def sweep_temperatures(Ts: np.ndarray, frac_plus_init: float | None = None):
    """Barrido en T para obtener |M|, χ, C y E promedio."""
    Ms, chis, Cs, Es = [], [], [], []
    rng_master = np.random.default_rng(7)
    for T in Ts:
        M_all, E_all = [], []
        for _ in range(SEEDS_PER_T):
            rng = np.random.default_rng(rng_master.integers(0, 2**31-1))
            spins0 = random_spins(L, frac_plus=frac_plus_init, rng=rng)
            res = metropolis_sweeps(spins0, T=T, sweeps=SWEEPS_EQ, burn_in=BURN_IN, rng=rng)
            M_all.append(res["M"]); E_all.append(res["E"])
        M_cat, E_cat = np.concatenate(M_all), np.concatenate(E_all)
        obs = observables_from_series(M_cat, E_cat, T=T)
        Ms.append(obs["M"]); chis.append(obs["chi"]); Cs.append(obs["C"]); Es.append(obs["E"])
    return np.array(Ms), np.array(chis), np.array(Cs), np.array(Es)

def run_dynamics_at_T(T: float, frac_plus_init: float | None = None):
    """Dinámica a T fija (serie temporal de |M| y E)."""
    rng = np.random.default_rng(123)
    spins0 = random_spins(L, frac_plus=frac_plus_init, rng=rng)
    res = metropolis_sweeps(spins0, T=T, sweeps=SWEEPS_DYN, burn_in=0, rng=rng)
    M_t, E_t = res["M"], res["E"]
    half = max(1, len(M_t)//2)
    return M_t, E_t, M_t[half:]

def balanced_ramping_h(L_local: int = None, T_fixed: float = None, h_max: float = 0.8, total_sweeps: int = 800):
    """Evolución desde 50/50 mientras h aumenta linealmente hasta h_max a T fija."""
    L_use = int(L_local or L)
    T_use = float(T_fixed if T_fixed is not None else TEMPERATURE)
    rng = np.random.default_rng(123)
    spins = random_spins(L_use, frac_plus=0.5, rng=rng)
    hs = np.linspace(0.0, h_max, total_sweeps + 1)
    mags = np.empty_like(hs)
    mags[0] = abs(spins.sum()) / (L_use * L_use)
    snap_steps = [0, total_sweeps//2, total_sweeps]
    snaps = {0: spins.copy()}

    def one_sweep(spins, T, h_local):
        Lx = spins.shape[0]
        for _ in range(Lx * Lx):
            i = rng.integers(0, Lx); j = rng.integers(0, Lx)
            nn = (spins[(i+1) % Lx, j] + spins[(i-1) % Lx, j] +
                  spins[i, (j+1) % Lx] + spins[i, (j-1) % Lx])
            de = 2.0 * spins[i, j] * (J * nn + h_local)
            if de <= 0.0 or rng.random() < np.exp(-de / (KB * T)):
                spins[i, j] *= -1
        return spins

    for s in range(1, total_sweeps + 1):
        spins = one_sweep(spins, T_use, hs[s])
        if s in snap_steps: snaps[s] = spins.copy()
        mags[s] = abs(spins.sum()) / (L_use * L_use)

    fig = plt.figure(figsize=(14.5, 8))
    gs = GridSpec(2, 3, height_ratios=[1.0, 0.7], hspace=0.25, wspace=0.15)
    for col, step in enumerate(snap_steps):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(snaps[step], cmap=SPIN_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        ax.axis("off"); ax.set_title(f"50/50 — $h = {hs[step]:.2f}$", fontsize=12)
    ax_bot = fig.add_subplot(gs[1, :])
    steps_arr = np.arange(total_sweeps + 1)
    ax_bot.plot(steps_arr, mags, label="|M|")
    ax_bot.set_xlabel("Barridos MC"); ax_bot.set_ylabel("Magnetización |M|")
    ax2 = ax_bot.twinx()
    ax2.plot(steps_arr, hs, linestyle="--", label="h")
    ax2.set_ylabel("h (campo externo)")
    ax_bot.set_title(f"Evolución 50/50 con rampa de h (T = {T_use:.2f})")
    ax_bot.grid(True, alpha=0.25)
    plt.tight_layout(); plt.show()

def realistic_ramping_h(
    L_local: int = None,
    T_fixed: float = None,
    h_max: float = 0.8,
    total_sweeps: int = 800,
    frac_plus_init: float = REALISTIC_FRACTION_WARM
):
    """Evolución desde estado sesgado (realista) mientras h aumenta linealmente a T fija."""
    L_use = int(L_local or L)
    T_use = float(T_fixed if T_fixed is not None else TEMPERATURE)
    rng = np.random.default_rng(321)
    spins = random_spins(L_use, frac_plus=frac_plus_init, rng=rng)
    hs = np.linspace(0.0, h_max, total_sweeps + 1)
    mags = np.empty_like(hs)
    mags[0] = abs(spins.sum()) / (L_use * L_use)
    snap_steps = [0, total_sweeps//2, total_sweeps]
    snaps = {0: spins.copy()}

    def one_sweep(spins, T, h_local):
        Lx = spins.shape[0]
        for _ in range(Lx * Lx):
            i = rng.integers(0, Lx); j = rng.integers(0, Lx)
            nn = (spins[(i+1) % Lx, j] + spins[(i-1) % Lx, j] +
                  spins[i, (j+1) % Lx] + spins[i, (j-1) % Lx])
            de = 2.0 * spins[i, j] * (J * nn + h_local)
            if de <= 0.0 or rng.random() < np.exp(-de / (KB * T)):
                spins[i, j] *= -1
        return spins

    for s in range(1, total_sweeps + 1):
        spins = one_sweep(spins, T_use, hs[s])
        if s in snap_steps: snaps[s] = spins.copy()
        mags[s] = abs(spins.sum()) / (L_use * L_use)

    fig = plt.figure(figsize=(14.5, 8))
    gs = GridSpec(2, 3, height_ratios=[1.0, 0.7], hspace=0.25, wspace=0.15)
    for col, step in enumerate(snap_steps):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(snaps[step], cmap=SPIN_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        ax.axis("off"); ax.set_title(f"Realista (sesgo {int(frac_plus_init*100)}%) — $h = {hs[step]:.2f}$", fontsize=12)
    ax_bot = fig.add_subplot(gs[1, :])
    steps_arr = np.arange(total_sweeps + 1)
    ax_bot.plot(steps_arr, mags, label="|M|")
    ax_bot.set_xlabel("Barridos MC"); ax_bot.set_ylabel("Magnetización |M|")
    ax2 = ax_bot.twinx()
    ax2.plot(steps_arr, hs, linestyle="--", label="h")
    ax2.set_ylabel("h (campo externo)")
    ax_bot.set_title(f"Evolución REALISTA con rampa de h (T = {T_use:.2f})")
    ax_bot.grid(True, alpha=0.25)
    plt.tight_layout(); plt.show()

def realistic_evolution_with_data(
    L_local: int = None,
    T_fixed: float = None,
    steps: int = None,
    h_max: float = 0.8,
    berkeley_path: str = None,
    rng_seed: int = 2025
):
    """Inicializa usando anomalía reciente (si hay archivo) y aumenta h a T fija."""
    L_use = int(L_local or L)
    T_use = float(T_fixed if T_fixed is not None else TEMPERATURE)
    steps_use = int(steps if steps is not None else V6_STEPS)

    recent_anom = recent_anomaly_from_berkeley(berkeley_path or BERKELEY_PATH)
    frac_plus0 = soft_map_anomaly_to_frac_plus(recent_anom)
    h_values = np.linspace(0.0, h_max, steps_use)
    rng = np.random.default_rng(rng_seed)
    spins = random_spins(L_use, frac_plus=frac_plus0, rng=rng)
    mags, snapshots = [], {}
    for k, h_local in enumerate(h_values):

        global H_FIELD
        old_h = H_FIELD
        H_FIELD = h_local
        try:
            res = metropolis_sweeps(
                spins, T=T_use,
                sweeps=800 if FAST else 2000,
                burn_in=300 if FAST else 1000,
                rng=np.random.default_rng(rng.integers(1, 2**31-1))
            )
        finally:
            H_FIELD = old_h
        mags.append(float(np.mean(res["M"])))
        spins = res["spins"]
        if k in (0, steps_use//2, steps_use-1):
            snapshots[k] = spins.copy()
    mags = np.array(mags)

    # Curva |M|(h)
    plt.figure(figsize=(9.5, 6))
    plt.plot(h_values, mags, marker="o")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Evolución Realista bajo Forzamiento Creciente (T = {T_use:.2f})")
    plt.xlabel("h (campo externo)"); plt.ylabel("|M|")
    plt.grid(True); plt.tight_layout(); plt.show()

    # Zona de mayor sensibilidad
    diffs = np.diff(mags) / np.diff(h_values) if len(h_values) > 1 else np.array([0.0])
    k_star = int(np.argmax(np.abs(diffs))) if diffs.size else 0
    h_star = h_values[min(k_star+1, len(h_values)-1)]
    m_star = mags[min(k_star+1, len(mags)-1)]

    plt.figure(figsize=(9.5, 6))
    plt.plot(h_values, mags, marker="o")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    band = 0.06 * h_max
    plt.axvspan(max(0.0, h_star-band), min(h_max, h_star+band), color="orange", alpha=0.15, label="Zona crítica")
    plt.annotate("Estado frío (sesgo inicial)", xy=(h_values[0], mags[0]),
                 xytext=(h_values[0]+0.1*h_max, mags[0]+0.15), arrowprops=dict(arrowstyle="->", lw=1.4))
    plt.annotate("Transición / alta sensibilidad", xy=(h_star, m_star),
                 xytext=(min(h_max, h_star+0.2*h_max), min(1.0, m_star+0.25)), arrowprops=dict(arrowstyle="->", lw=1.4))
    plt.annotate("Nuevo equilibrio (cálido)", xy=(h_values[-1], mags[-1]),
                 xytext=(h_values[-1]-0.45*h_max, max(0.05, mags[-1]-0.25)), arrowprops=dict(arrowstyle="->", lw=1.4))
    plt.title("Evolución Realista (Anotada)")
    plt.xlabel("h (campo externo)"); plt.ylabel("|M|")
    plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout(); plt.show()

    if snapshots:
        plt.figure(figsize=(12, 4))
        keys = sorted(snapshots.keys()); titles = ["Inicio", "Medio", "Final"]
        for idx, (k_snap, ttl) in enumerate(zip(keys, titles), start=1):
            plt.subplot(1, 3, idx)
            plt.imshow(snapshots[k_snap], vmin=-1, vmax=1, cmap=SPIN_CMAP, interpolation="nearest")
            plt.title(f"{ttl}\nh = {h_values[k_snap]:.2f}"); plt.axis("off")
        plt.suptitle("Configuraciones durante la Evolución", fontsize=14, weight="bold")
        plt.tight_layout(); plt.show()

    return {"h": h_values, "M": mags, "anom_recent": recent_anom, "frac_plus0": frac_plus0}

def main():
    global H_FIELD

    h_data = 0.0
    clima_info = None
    if USE_BERKELEY_FIELD and os.path.exists(BERKELEY_PATH):
        try:
            h_data = h_from_berkeley(BERKELEY_PATH)
            H_FIELD = H_FIELD + h_data 
            clima_info = climate_series_from_berkeley(BERKELEY_PATH)
        except Exception:
            clima_info = climate_series_from_berkeley(BERKELEY_PATH)
    else:
        if pd is not None and os.path.exists(BERKELEY_PATH):
            clima_info = climate_series_from_berkeley(BERKELEY_PATH)

    rng0 = np.random.default_rng(1)
    spins_balanced = random_spins(L, frac_plus=None, rng=rng0)
    plot_initial(spins_balanced, frac_plus=None, clima_series=clima_info, title_suffix=" - Guatemala")

    Ts = np.arange(T_MIN, T_MAX + 1e-9, T_STEP)
    Ms, chis, Cs, Es = sweep_temperatures(Ts, frac_plus_init=None)
    plot_thermo(Ts, Ms, chis, Cs, Es, Tc_guess=2.269, title_suffix="")

    M_t, E_t, M_eq = run_dynamics_at_T(TEMPERATURE, frac_plus_init=None)
    Ts_zoom = np.arange(1.75, 2.40 + 1e-9, 0.05)
    Ms_zoom, _, _, _ = sweep_temperatures(Ts_zoom, frac_plus_init=None)
    plot_dynamics_and_critical(TEMPERATURE, M_t, E_t, M_eq, Ts_zoom, Ms_zoom, Tc_guess=2.269, title_suffix="")

    balanced_ramping_h(L_local=L, T_fixed=TEMPERATURE, h_max=0.8, total_sweeps=SWEEPS_DYN)
    realistic_ramping_h(L_local=L, T_fixed=TEMPERATURE, h_max=0.8, total_sweeps=SWEEPS_DYN,
                        frac_plus_init=REALISTIC_FRACTION_WARM)

    if REALISTIC_INIT:
        spins_real = random_spins(L, frac_plus=REALISTIC_FRACTION_WARM, rng=np.random.default_rng(2))
        plot_initial(spins_real, frac_plus=REALISTIC_FRACTION_WARM, clima_series=clima_info,
                     title_suffix=" - Estado REALISTA")
        Ms_r, chis_r, Cs_r, Es_r = sweep_temperatures(Ts, frac_plus_init=REALISTIC_FRACTION_WARM)
        plot_thermo(Ts, Ms_r, chis_r, Cs_r, Es_r, Tc_guess=2.269, title_suffix=" - Sistema REALISTA")

    realistic_evolution_with_data(L_local=L, T_fixed=TEMPERATURE, steps=V6_STEPS, h_max=0.8,
                                  berkeley_path=BERKELEY_PATH, rng_seed=2025)

if __name__ == "__main__":
    main()
