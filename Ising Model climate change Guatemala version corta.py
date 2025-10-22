import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import os
from typing import Optional, Dict, Tuple

L = 32  # Tamaño reducido del grid
SWEEPS_EQ = 400   # Barridos reducidos para equilibrio
SWEEPS_DYN = 400  # Barridos reducidos para dinámica
BURN_IN = 200     # Burn-in reducido

T_MIN, T_MAX, T_STEP = 1.7, 3.2, 0.3
T_CRITICAL = 2.27

SPIN_CMAP = ListedColormap(["blue", "red"])

class IsingSimulator:
    def __init__(self, L: int, J: float = 1.0, kB: float = 1.0):
        self.L = L
        self.J = J
        self.kB = kB
        self.rng = np.random.default_rng(42)
        
    def random_spins(self, frac_plus: float = 0.5) -> np.ndarray:
        """Crea configuración inicial de espines"""
        n = self.L * self.L
        k = int(round(frac_plus * n))
        arr = np.array([1]*k + [-1]*(n-k), dtype=int)
        self.rng.shuffle(arr)
        return arr.reshape(self.L, self.L)
    
    def metropolis_step(self, spins: np.ndarray, T: float, h: float = 0.0) -> np.ndarray:
        """Un paso de Metropolis optimizado"""
        Lx = self.L
        i = self.rng.integers(0, Lx, size=Lx*Lx)
        j = self.rng.integers(0, Lx, size=Lx*Lx)
        
        for idx in range(Lx*Lx):
            ii, jj = i[idx], j[idx]
            
            nn = (spins[(ii+1) % Lx, jj] + spins[(ii-1) % Lx, jj] +
                  spins[ii, (jj+1) % Lx] + spins[ii, (jj-1) % Lx])
            
            dE = 2.0 * spins[ii, jj] * (self.J * nn + h)
            
            if dE <= 0.0 or self.rng.random() < np.exp(-dE / (self.kB * T)):
                spins[ii, jj] *= -1
                
        return spins
    
    def simulate(self, spins: np.ndarray, T: float, h: float = 0.0, 
                 sweeps: int = 500, burn_in: int = 200) -> Dict[str, np.ndarray]:
        """Simulación completa optimizada"""
        n = self.L * self.L
        M_series, E_series = [], []

        for _ in range(burn_in):
            spins = self.metropolis_step(spins, T, h)

        for sweep in range(sweeps):
            spins = self.metropolis_step(spins, T, h)
            
            if sweep % 5 == 0:
                M = np.sum(spins) / n
                E = self.calculate_energy(spins, h) / n
                M_series.append(abs(M))
                E_series.append(E)
        
        return {
            "M": np.array(M_series),
            "E": np.array(E_series), 
            "spins": spins,
            "M_final": np.mean(M_series[-50:]) if len(M_series) > 50 else np.mean(M_series)
        }
    
    def calculate_energy(self, spins: np.ndarray, h: float) -> float:
        """Calcula energía total optimizada"""
        E = -self.J * np.sum(spins * np.roll(spins, -1, axis=0))
        E -= self.J * np.sum(spins * np.roll(spins, -1, axis=1))
        E -= h * np.sum(spins)
        return E
    
    def sweep_temperatures(self, T_range: np.ndarray, frac_plus: float = 0.5, 
                          h: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Barrido de temperaturas optimizado"""
        Ms, chis, Cs, Es = [], [], [], []
        
        for T in T_range:
            spins = self.random_spins(frac_plus)
            result = self.simulate(spins, T, h, SWEEPS_EQ, BURN_IN)
            
            M, E = result["M"], result["E"]
            
            M_mean = np.mean(M)
            M_var = np.var(M)
            E_mean = np.mean(E) 
            E_var = np.var(E)
            
            chi = M_var / (self.kB * T) if T > 0 else 0
            C = E_var / (self.kB * T**2) if T > 0 else 0
            
            Ms.append(M_mean)
            chis.append(chi)
            Cs.append(C)
            Es.append(E_mean)
            
            print(f"T = {T:.2f}, M = {M_mean:.3f}")
        
        return np.array(Ms), np.array(chis), np.array(Cs), np.array(Es)

class IsingVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
        
    def plot_initial_config(self, spins: np.ndarray, clima_data: Optional[dict] = None, 
                           title: str = "Configuración Inicial"):
        """Ventana 1: Configuración inicial"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle(title, fontsize=16, weight="bold")
        
        # Datos climáticos
        if clima_data and 'years' in clima_data:
            years, anom = clima_data['years'], clima_data['anom']
            ax1.plot(years, anom, 'r-', linewidth=2, label='Anomalía')
            if 'unc' in clima_data:
                ax1.fill_between(years, anom-clima_data['unc'], anom+clima_data['unc'], 
                               alpha=0.3, label='Incertidumbre')
            ax1.set_title("Datos Climáticos Guatemala")
            ax1.set_xlabel("Año"); ax1.set_ylabel("Anomalía (°C)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.axis('off')
            ax1.text(0.5, 0.5, "Datos climáticos\nno disponibles", 
                    ha='center', va='center', transform=ax1.transAxes)

        im = ax2.imshow(spins, cmap=SPIN_CMAP, vmin=-1, vmax=1)
        ax2.set_title("Configuración de Espines")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")
        plt.colorbar(im, ax=ax2, shrink=0.8).set_label("Espín: -1 (frío), +1 (cálido)")
        
        # Histograma
        vals = spins.ravel()
        counts = [np.sum(vals == -1), np.sum(vals == 1)]
        ax3.bar(['Frío (-1)', 'Cálido (+1)'], counts, color=['blue', 'red'], alpha=0.7)
        ax3.set_title("Distribución de Espines")
        ax3.set_ylabel("Recuento")
        
        # Información del sistema
        ax4.axis('off')
        frac_calido = np.sum(vals == 1) / len(vals)
        info_text = f"""SISTEMA ISING 2D
• Tamaño: {spins.shape[0]}×{spins.shape[1]}
• Espines cálidos: {frac_calido*100:.1f}%
• Magnetización: {np.mean(vals):.3f}
• Hamiltoniano: H = -J∑sᵢsⱼ"""
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.show()
    
    def plot_thermodynamics(self, T_range: np.ndarray, Ms: np.ndarray, chis: np.ndarray, 
                           Cs: np.ndarray, Es: np.ndarray, Tc: float = 2.27):
        """Ventana 2: Propiedades termodinámicas"""
        fig, axs = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle("Transición de Fase - Propiedades Termodinámicas", fontsize=16, weight="bold")
        
        plots_data = [
            (Ms, 'Magnetización |M|', 'blue', 'o'),
            (chis, 'Susceptibilidad χ', 'red', 'o'), 
            (Cs, 'Capacidad Calorífica C', 'green', '^'),
            (Es, 'Energía ⟨E⟩', 'purple', 's')
        ]
        
        for idx, (data, title, color, marker) in enumerate(plots_data):
            ax = axs[idx//2, idx%2]
            ax.plot(T_range, data, marker=marker, color=color, linewidth=2)
            ax.axvline(Tc, linestyle='--', color='black', alpha=0.7, label=f'Tc = {Tc:.2f}')
            ax.set_title(title)
            ax.set_xlabel("T (J/kB)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_dynamics(self, M_series: np.ndarray, E_series: np.ndarray, T: float):
        """Ventana 3: Dinámica del sistema"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Dinámica del Sistema - T = {T:.2f}", fontsize=16, weight="bold")
        
        # Evolución temporal
        ax1.plot(M_series, 'b-', label='Magnetización |M|', linewidth=2)
        E_norm = (E_series - E_series.min()) / (E_series.max() - E_series.min() + 1e-12)
        ax1.plot(E_norm, 'r-', alpha=0.7, label='Energía (norm.)')
        ax1.set_xlabel("Pasos de Monte Carlo")
        ax1.set_ylabel("Magnitud")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Evolución Temporal")
        
        # Histograma de magnetización
        ax2.hist(M_series, bins=15, color='blue', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(M_series), color='red', linestyle='--', 
                   label=f'⟨M⟩ = {np.mean(M_series):.3f}')
        ax2.set_xlabel("Magnetización")
        ax2.set_ylabel("Frecuencia")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Distribución de Magnetización")
        
        plt.tight_layout()
        plt.show()
    
    def plot_field_evolution(self, h_values: np.ndarray, M_values: np.ndarray, 
                           snapshots: dict, T: float):
        """Ventana 4: Evolución con campo externo"""
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 4, height_ratios=[1.2, 0.8])
        
        # Snapshots
        snapshot_positions = [(0, 0), (0, 1), (0, 2)]
        titles = ['Inicio (h=0)', 'Transición', 'Final']
        
        for idx, (pos, title) in enumerate(zip(snapshot_positions, titles)):
            if idx in snapshots:
                ax = fig.add_subplot(gs[pos])
                ax.imshow(snapshots[idx], cmap=SPIN_CMAP, vmin=-1, vmax=1)
                ax.set_title(f"{title}\nh = {h_values[idx]:.2f}")
                ax.axis('off')
        
        # Evolución de magnetización
        ax_main = fig.add_subplot(gs[1, :])
        ax_main.plot(h_values, M_values, 'bo-', linewidth=2, markersize=4, label='|M|')
        ax_main.set_xlabel("Campo externo h")
        ax_main.set_ylabel("Magnetización |M|", color='blue')
        ax_main.tick_params(axis='y', labelcolor='blue')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_title(f"Evolución con Campo Externo - T = {T:.2f}")
        
        # Marcar zona crítica
        if len(M_values) > 1:
            derivadas = np.abs(np.diff(M_values) / np.diff(h_values))
            if len(derivadas) > 0:
                idx_critico = np.argmax(derivadas)
                h_critico = h_values[idx_critico + 1]
                ax_main.axvspan(h_critico-0.1, h_critico+0.1, alpha=0.2, color='red', 
                              label='Zona crítica')
        
        ax_main.legend()
        plt.tight_layout()
        plt.show()

def simulate_field_ramp(simulator: IsingSimulator, T: float, h_max: float = 0.8, 
                       steps: int = 12, frac_plus: float = 0.5) -> dict:
    """Simula evolución con campo externo creciente"""
    h_values = np.linspace(0, h_max, steps)
    M_values = []
    snapshots = {}
    
    spins = simulator.random_spins(frac_plus)
    
    for idx, h in enumerate(h_values):
        result = simulator.simulate(spins.copy(), T, h, SWEEPS_DYN//2, BURN_IN//2)
        M_values.append(result["M_final"])
        
        # Guardar snapshots clave
        if idx in [0, steps//2, steps-1]:
            snapshots[idx] = result["spins"]
    
    return {"h": h_values, "M": M_values, "snapshots": snapshots}

def main():
    print("🚀 INICIANDO ANÁLISIS ISING OPTIMIZADO - GUATEMALA")
    print("=" * 50)
    
    simulator = IsingSimulator(L)
    visualizer = IsingVisualizer()
    
    clima_data = {
        'years': np.arange(1990, 2024),
        'anom': 0.01 * (np.arange(1990, 2024) - 1990) + 0.1 * np.random.randn(34),
        'unc': 0.05 * np.ones(34)
    }

    print("\n📊 VENTANA 1: Configuración inicial")
    spins_balanceados = simulator.random_spins(0.5)
    visualizer.plot_initial_config(spins_balanceados, clima_data, "Estado Balanceado 50/50")

    print("\n🔥 VENTANA 2: Análisis termodinámico")
    T_range = np.arange(T_MIN, T_MAX + 0.1, T_STEP)
    Ms, chis, Cs, Es = simulator.sweep_temperatures(T_range)
    visualizer.plot_thermodynamics(T_range, Ms, chis, Cs, Es, T_CRITICAL)

    print("\n🔄 VENTANA 3: Dinámica del sistema")
    spins_dinamica = simulator.random_spins(0.5)
    resultado_dinamica = simulator.simulate(spins_dinamica, T_CRITICAL, sweeps=SWEEPS_DYN)
    visualizer.plot_dynamics(resultado_dinamica["M"], resultado_dinamica["E"], T_CRITICAL)

    print("\n📈 VENTANA 4: Evolución con forzamiento externo")
    resultado_rampa = simulate_field_ramp(simulator, T=2.7, frac_plus=0.5)
    visualizer.plot_field_evolution(
        resultado_rampa["h"], resultado_rampa["M"], 
        resultado_rampa["snapshots"], T=2.7
    )

    print("\n🌡️ VENTANA 5: Sistema realista (60% cálido)")
    spins_realista = simulator.random_spins(0.6)
    visualizer.plot_initial_config(spins_realista, clima_data, "Estado Realista 60/40")
    
    Ms_real, chis_real, Cs_real, Es_real = simulator.sweep_temperatures(T_range, 0.6)
    visualizer.plot_thermodynamics(T_range, Ms_real, chis_real, Cs_real, Es_real, T_CRITICAL)

    print("\n🌍 VENTANA 6: Evolución realista con calentamiento")
    resultado_realista = simulate_field_ramp(simulator, T=2.27, frac_plus=0.6)
    visualizer.plot_field_evolution(
        resultado_realista["h"], resultado_realista["M"],
        resultado_realista["snapshots"], T=2.27
    )
    
    print("\n✅ ANÁLISIS COMPLETADO")
    print("=" * 50)

if __name__ == "__main__":
    main()