#!/usr/bin/env python3
"""
Λ³-Dynamics Tensile Test Simulator v3
======================================

統一δ理論に基づく実装

核心原理:
  δ_total = δ_thermal + δ_mech
  
  δ_thermal = √(kT/Mω²) / r_nn     # 熱揺らぎ（全原子共通）
  δ_mech = σ_local / E(T)          # 弾性ひずみのみ！（塑性は含まない）
  
  塑性変形:
    - マクロでは ε_plastic 蓄積（試験片が伸びる）
    - 原子では転位通過 → 格子リセット → δ_mech 解放
    - 解放されたエネルギーは熱に変換

相図:
  δ < 0.01: Hooke（完全弾性）
  δ < 0.03: 非線形弾性
  δ < 0.05: 降伏域（転位増殖）
  δ < 0.10: 塑性流動
  δ ≥ 0.10: Lindemann（破壊/融解）

応力集中:
  - 空孔周り: K_t = 1 + 2√(a/r)
  - 表面/コーナー: K_t from Z_eff
  - テール分布で0.3%がδ_L超え → 降伏開始

Author: Tamaki & Masamichi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
import math

# 物理定数
k_B = 1.380649e-23  # Boltzmann定数 [J/K]
u_kg = 1.66053906660e-27  # 原子質量単位 [kg]


class DeformationPhase(Enum):
    """変形相（統一δ理論）"""
    HOOKE = "HOOKE"              # δ < 0.01: 完全弾性
    NONLINEAR = "NONLINEAR"      # δ < 0.03: 非線形弾性
    YIELD = "YIELD"              # δ < 0.05: 降伏域
    PLASTIC = "PLASTIC"          # δ < 0.10: 塑性流動
    FAILURE = "FAILURE"          # δ ≥ 0.10: Lindemann超え


@dataclass
class MaterialData:
    """材料データ"""
    name: str
    structure: str       # "FCC", "BCC", "HCP"
    Z_bulk: int          # バルク配位数
    a_300K: float        # 格子定数 [m] @ 300K
    alpha: float         # 熱膨張係数 [1/K]
    E0: float            # ヤング率 [Pa] @ 300K
    nu: float            # ポアソン比
    T_melt: float        # 融点 [K]
    M_amu: float         # 原子量 [amu]
    rho: float           # 密度 [kg/m³]
    delta_L: float       # Lindemann定数
    sigma_y: float       # 降伏応力 [Pa]
    # Λ³熱軟化パラメータ
    lambda_base: float = 30.0
    kappa: float = 2.0
    
    @classmethod
    def FCC_Cu(cls):
        """FCC銅"""
        return cls(
            name="FCC-Cu",
            structure="FCC",
            Z_bulk=12,
            a_300K=3.61e-10,
            alpha=1.7e-5,
            E0=130e9,
            nu=0.34,
            T_melt=1357,
            M_amu=63.546,
            rho=8960,
            delta_L=0.10,
            sigma_y=122e6,  # 122 MPa
            lambda_base=26.3,
            kappa=1.713,
        )
    
    @classmethod
    def BCC_Fe(cls):
        """BCC鉄"""
        return cls(
            name="BCC-Fe",
            structure="BCC",
            Z_bulk=8,
            a_300K=2.87e-10,
            alpha=1.5e-5,
            E0=210e9,
            nu=0.29,
            T_melt=1811,
            M_amu=55.845,
            rho=7870,
            delta_L=0.18,
            sigma_y=250e6,  # 250 MPa
            lambda_base=49.2,
            kappa=0.573,
        )


class UnifiedDeltaEngine:
    """
    統一δ理論エンジン
    
    δ = δ_thermal + δ_mech
    
    塑性ひずみはδに寄与しない（転位通過で解放）
    """
    
    # 相境界（δ値）
    DELTA_HOOKE = 0.01
    DELTA_NONLINEAR = 0.03
    DELTA_YIELD = 0.05
    DELTA_PLASTIC = 0.10  # = δ_L (Lindemann)
    
    def __init__(self, material: MaterialData):
        self.mat = material
        self.M = material.M_amu * u_kg
        
        # 室温弾性定数
        self.G0 = material.E0 / (2.0 * (1.0 + material.nu))
        self.K0 = material.E0 / (3.0 * (1.0 - 2.0 * material.nu))
        
        print("="*60)
        print("Unified δ-Theory Engine")
        print("="*60)
        print(f"Material: {material.name}")
        print(f"E₀ = {material.E0/1e9:.1f} GPa")
        print(f"σ_y = {material.sigma_y/1e6:.1f} MPa")
        print(f"δ_L = {material.delta_L}")
        print(f"T_melt = {material.T_melt} K")
    
    # ========================================
    # 温度依存パラメータ
    # ========================================
    
    def lattice_constant(self, T: float) -> float:
        """格子定数 a(T) [m]"""
        return self.mat.a_300K * (1.0 + self.mat.alpha * (T - 300.0))
    
    def nearest_neighbor_distance(self, T: float) -> float:
        """最近接原子間距離 r_nn(T) [m]"""
        a = self.lattice_constant(T)
        if self.mat.structure == "BCC":
            return a * math.sqrt(3) / 2
        elif self.mat.structure == "FCC":
            return a / math.sqrt(2)
        return a / math.sqrt(2)
    
    def thermal_softening(self, T: float) -> float:
        """
        Λ³熱軟化 E(T)/E₀
        
        E(T)/E₀ = exp[-λ_eff × α × ΔT]
        λ_eff = λ_base × (1 + κ × ΔT/1000)
        """
        T_ref = 293.0
        if T <= T_ref:
            return 1.0
        
        delta_T = T - T_ref
        lambda_eff = self.mat.lambda_base * (1.0 + self.mat.kappa * delta_T / 1000.0)
        return math.exp(-lambda_eff * self.mat.alpha * delta_T)
    
    def youngs_modulus(self, T: float) -> float:
        """温度依存ヤング率 E(T) [Pa]"""
        return self.mat.E0 * self.thermal_softening(T)
    
    def shear_modulus(self, T: float) -> float:
        """温度依存剛性率 G(T) [Pa]"""
        return self.G0 * self.thermal_softening(T)
    
    # ========================================
    # δ計算（統一理論）
    # ========================================
    
    def delta_thermal(self, T: float) -> float:
        """
        熱的δ成分
        
        δ_thermal = √⟨u²⟩ / r_nn
        ⟨u²⟩ = (k_B T / M) × ⟨1/ω²⟩
        """
        if T <= 0:
            return 0.0
        
        # Debye-Waller計算
        G = self.shear_modulus(T)
        K = self.K0 * (1.0 - 0.3 * (T / self.mat.T_melt)**2)  # 体積弾性率
        
        a = self.lattice_constant(T)
        if self.mat.structure == "FCC":
            n = 4.0 / (a**3)
        elif self.mat.structure == "BCC":
            n = 2.0 / (a**3)
        else:
            n = 4.0 / (a**3)
        
        rho = n * self.M
        
        # 音速
        v_t = math.sqrt(G / rho) if G > 0 else 1.0
        v_l = math.sqrt((K + 4.0*G/3.0) / rho) if (K + 4.0*G/3.0) > 0 else 1.0
        
        # Debye波数
        k_D = (6.0 * math.pi**2 * n) ** (1.0/3.0)
        
        # ⟨1/ω²⟩
        inv_omega2 = (1.0 / (3.0 * k_D**2)) * (2.0/v_t**2 + 1.0/v_l**2)
        
        # ⟨u²⟩
        u2 = (k_B * T / self.M) * inv_omega2
        
        # δ_thermal
        r_nn = self.nearest_neighbor_distance(T)
        return math.sqrt(u2) / r_nn
    
    def delta_mechanical(self, sigma_local: float, T: float) -> float:
        """
        機械的δ成分（弾性ひずみのみ！）
        
        δ_mech = σ_local / E(T)
        
        これがHookeの法則の原子スケール表現
        """
        E_T = self.youngs_modulus(T)
        if E_T <= 0:
            return 0.0
        return abs(sigma_local) / E_T
    
    def delta_total(self, sigma_local: float, T: float) -> float:
        """
        合計δ
        
        δ_total = δ_thermal + δ_mech
        """
        d_th = self.delta_thermal(T)
        d_mech = self.delta_mechanical(sigma_local, T)
        return d_th + d_mech
    
    # ========================================
    # 相判定
    # ========================================
    
    def determine_phase(self, delta: float) -> DeformationPhase:
        """δから変形相を判定"""
        if delta < self.DELTA_HOOKE:
            return DeformationPhase.HOOKE
        elif delta < self.DELTA_NONLINEAR:
            return DeformationPhase.NONLINEAR
        elif delta < self.DELTA_YIELD:
            return DeformationPhase.YIELD
        elif delta < self.DELTA_PLASTIC:
            return DeformationPhase.PLASTIC
        else:
            return DeformationPhase.FAILURE
    
    def is_yielded(self, delta: float) -> bool:
        """降伏したか（δ > δ_yield）"""
        return delta >= self.DELTA_YIELD
    
    def is_failed(self, delta: float) -> bool:
        """破壊したか（δ ≥ δ_L）"""
        return delta >= self.mat.delta_L


class StressConcentrationField:
    """
    応力集中場
    
    K_t = σ_local / σ_applied
    
    応力集中源:
      1. 空孔からの距離: K_t = 1 + A/√r （最も強い！）
      2. 配位数低下: K_t × (Z_bulk / Z_eff)
      3. 表面/境界: 追加補正
    
    これにより「テール分布」が生成される！
    平均δ ≈ 0.03 でも、空孔近傍で δ > δ_L となる
    """
    
    def __init__(self, 
                 N: int,
                 Z_bulk: int = 12,
                 vacancy_fraction: float = 0.02):
        """
        Args:
            N: 格子サイズ（N³）
            Z_bulk: バルク配位数
            vacancy_fraction: 空孔率
        """
        self.N = N
        self.Z_bulk = Z_bulk
        
        # 格子と配位数を初期化
        self.lattice = np.ones((N, N, N), dtype=bool)
        self.Z = np.ones((N, N, N), dtype=int) * Z_bulk
        
        # 空孔を導入
        self.vacancy_positions = []
        self._introduce_vacancies(vacancy_fraction)
        
        # 表面を検出
        self._detect_surfaces()
        
        # 空孔からの距離場を計算（応力集中の主要因！）
        self._compute_distance_to_vacancies()
        
        n_active = np.sum(self.lattice)
        print(f"\nStress Concentration Field:")
        print(f"  Lattice: {N}³ = {N**3}")
        print(f"  Active sites: {n_active}")
        print(f"  Vacancies: {len(self.vacancy_positions)}")
        print(f"  Z range: [{self.Z[self.lattice].min()}, {self.Z[self.lattice].max()}]")
    
    def _introduce_vacancies(self, fraction: float):
        """空孔を導入し、周囲のZを更新"""
        N = self.N
        n_vac = int(N**3 * fraction)
        
        np.random.seed(42)  # 再現性
        vac_pos = np.random.choice(N**3, n_vac, replace=False)
        
        for pos in vac_pos:
            i = pos // (N*N)
            j = (pos % (N*N)) // N
            k = pos % N
            
            self.lattice[i, j, k] = False
            self.vacancy_positions.append((i, j, k))
            
            # 隣接原子のZを減少
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if di == 0 and dj == 0 and dk == 0:
                            continue
                        ni, nj, nk = i+di, j+dj, k+dk
                        if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                            if self.lattice[ni, nj, nk]:
                                self.Z[ni, nj, nk] = max(0, self.Z[ni, nj, nk] - 1)
    
    def _detect_surfaces(self):
        """表面原子を検出し、Zを調整"""
        N = self.N
        
        # 境界面でZを減少
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if not self.lattice[i, j, k]:
                        continue
                    
                    # 境界に接している数をカウント
                    n_boundary = 0
                    if i == 0 or i == N-1: n_boundary += 1
                    if j == 0 or j == N-1: n_boundary += 1
                    if k == 0 or k == N-1: n_boundary += 1
                    
                    # 境界接触分だけZ減少
                    self.Z[i, j, k] = max(1, self.Z[i, j, k] - n_boundary)
    
    def _compute_distance_to_vacancies(self):
        """
        各格子点から最近傍の空孔への距離を計算
        
        これが応力集中の主要因！
        """
        N = self.N
        self.dist_to_vacancy = np.full((N, N, N), np.inf)
        
        if len(self.vacancy_positions) == 0:
            return
        
        # KDTreeで効率的に最近傍探索
        vac_coords = np.array(self.vacancy_positions)
        tree = cKDTree(vac_coords)
        
        # 全アクティブ点について最近傍空孔への距離
        active_indices = np.where(self.lattice)
        active_coords = np.column_stack(active_indices)
        
        distances, _ = tree.query(active_coords, k=1)
        
        for idx, (i, j, k) in enumerate(zip(*active_indices)):
            self.dist_to_vacancy[i, j, k] = distances[idx]
        
        # 距離分布を出力
        dist_active = self.dist_to_vacancy[self.lattice]
        print(f"  Distance to vacancy: min={dist_active.min():.2f}, "
              f"max={dist_active.max():.2f}, mean={dist_active.mean():.2f}")
    
    def stress_concentration_factor(self) -> np.ndarray:
        """
        応力集中係数 K_t を計算
        
        K_t = K_t_distance × K_t_coordination
        
        K_t_distance = 1 + A / √(r + r_min)
          → 空孔近傍（r→0）で K_t → 大
          → A = 20, r_min = 0.1 で調整
        
        K_t_coordination = Z_bulk / Z_eff
          → Z低下でさらに増幅
        """
        N = self.N
        K_t = np.ones((N, N, N))
        
        mask = self.lattice & (self.Z > 0)
        
        # 1. 距離ベースの応力集中（主要！）
        # K_t = 1 + A / √(r + r_min)
        # 実験との整合: σ_y で 0.2〜0.5% が δ_L 超え
        A = 40.0  # 強度パラメータ
        r_min = 0.05  # 最小距離
        K_t_max = 100.0  # 上限
        
        r = self.dist_to_vacancy[mask]
        K_t_dist = 1.0 + A / np.sqrt(r + r_min)
        
        # ★ 空孔隣接サイト（r ≤ 1.5）に追加ブースト
        # 実験: 空孔直隣は応力集中 5〜10倍
        neighbor_boost = np.where(r <= 1.5, 2.0, 1.0)  # 隣接は2倍ブースト
        K_t_dist = K_t_dist * neighbor_boost
        
        K_t_dist = np.minimum(K_t_dist, K_t_max)
        
        # 2. 配位数ベースの補正
        K_t_coord = self.Z_bulk / self.Z[mask]
        
        # 3. 複合効果（掛け算ではなく、大きい方を採用 + 補正）
        # K_t = max(K_t_dist, K_t_coord) × (1 + 0.1 × min(K_t_dist, K_t_coord))
        K_t[mask] = np.maximum(K_t_dist, K_t_coord) * (1.0 + 0.1 * np.minimum(K_t_dist, K_t_coord))
        
        return K_t
    
    def local_stress(self, sigma_applied: float) -> np.ndarray:
        """
        局所応力場 σ_local = K_t × σ_applied
        """
        K_t = self.stress_concentration_factor()
        return K_t * sigma_applied


class TensileTestV3:
    """
    統一δ理論に基づく引張試験シミュレータ
    
    特徴:
      - δ = δ_thermal + δ_mech（弾性のみ）
      - 応力集中による不均一δ分布
      - テール分布で局所的Lindemann超え
      - 降伏 = テールがδ_L超え → 転位源活性化
    """
    
    def __init__(self,
                 N: int = 50,
                 material: MaterialData = None,
                 vacancy_fraction: float = 0.02):
        """
        Args:
            N: 格子サイズ（N³）
            material: 材料データ
            vacancy_fraction: 空孔率
        """
        self.material = material or MaterialData.FCC_Cu()
        self.engine = UnifiedDeltaEngine(self.material)
        self.field = StressConcentrationField(
            N, 
            Z_bulk=self.material.Z_bulk,
            vacancy_fraction=vacancy_fraction
        )
        self.N = N
        
        # 状態変数
        self.melted = np.zeros((N, N, N), dtype=bool)  # 崩壊した格子点
        self.plastic_strain = 0.0  # マクロ塑性ひずみ（蓄積）
        
        # 履歴
        self.history = []
    
    def run_test(self,
                 sigma_max: float = None,
                 n_steps: int = 30,
                 T: float = 300.0) -> List[dict]:
        """
        引張試験を実行
        
        Args:
            sigma_max: 最大応力 [Pa]（Noneなら降伏応力）
            n_steps: ステップ数
            T: 温度 [K]
        
        Returns:
            各ステップの結果リスト
        """
        if sigma_max is None:
            sigma_max = self.material.sigma_y
        
        sigma_steps = np.linspace(0, sigma_max, n_steps)
        
        print("\n" + "="*80)
        print("TENSILE TEST (Unified δ-Theory)")
        print("="*80)
        print(f"Material: {self.material.name}")
        print(f"Temperature: {T} K")
        print(f"σ_max: {sigma_max/1e6:.1f} MPa")
        print(f"σ_y: {self.material.sigma_y/1e6:.1f} MPa")
        print("-"*80)
        
        # 熱的δ（温度で決まる、全原子共通）
        delta_thermal = self.engine.delta_thermal(T)
        print(f"δ_thermal(T={T}K) = {delta_thermal:.6f}")
        
        E_T = self.engine.youngs_modulus(T)
        print(f"E(T) = {E_T/1e9:.1f} GPa")
        
        print("-"*80)
        print(f"{'Step':<6} {'σ(MPa)':<10} {'δ_max':<10} {'δ_mean':<10} "
              f"{'Yield%':<10} {'Fail%':<10} {'Phase':<15}")
        print("-"*80)
        
        results = []
        
        for step, sigma_app in enumerate(sigma_steps):
            result = self._run_step(sigma_app, T, delta_thermal, E_T)
            results.append(result)
            
            # 出力
            print(f"{step:<6} {sigma_app/1e6:<10.1f} {result['delta_max']:<10.4f} "
                  f"{result['delta_mean']:<10.4f} {result['yield_frac']*100:<10.2f} "
                  f"{result['fail_frac']*100:<10.2f} {result['dominant_phase']:<15}")
            
            # 50%以上が破壊したら終了
            if result['fail_frac'] > 0.5:
                print(f"\n*** FRACTURE at σ = {sigma_app/1e6:.1f} MPa ***")
                break
        
        self.history = results
        return results
    
    def _run_step(self, 
                  sigma_app: float, 
                  T: float,
                  delta_thermal: float,
                  E_T: float) -> dict:
        """1ステップを実行"""
        N = self.N
        
        # 1. 局所応力場
        sigma_local = self.field.local_stress(sigma_app)
        
        # 2. 機械的δ（弾性ひずみ）
        # δ_mech = σ_local / E(T)
        delta_mech = np.zeros((N, N, N))
        mask = self.field.lattice & ~self.melted
        delta_mech[mask] = sigma_local[mask] / E_T
        
        # 3. 合計δ
        delta_total = np.zeros((N, N, N))
        delta_total[mask] = delta_thermal + delta_mech[mask]
        
        # 4. 相判定
        phases = np.full((N, N, N), DeformationPhase.HOOKE, dtype=object)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if mask[i, j, k]:
                        phases[i, j, k] = self.engine.determine_phase(delta_total[i, j, k])
        
        # 5. 新しく破壊した格子点
        newly_failed = (delta_total >= self.material.delta_L) & mask & ~self.melted
        n_new_fail = np.sum(newly_failed)
        
        if n_new_fail > 0:
            self.melted |= newly_failed
            
            # カスケード: 破壊した点の隣接原子のZを減少
            self._cascade_Z_reduction(newly_failed)
        
        # 6. 統計
        active_mask = self.field.lattice & ~self.melted
        n_active = np.sum(active_mask)
        
        if n_active > 0:
            delta_active = delta_total[active_mask]
            delta_max = np.max(delta_active)
            delta_mean = np.mean(delta_active)
            
            # 各相の割合
            phase_counts = {}
            for phase in DeformationPhase:
                phase_counts[phase] = np.sum(phases[active_mask] == phase)
            
            yield_frac = np.sum(delta_active >= self.engine.DELTA_YIELD) / n_active
            fail_frac = np.sum(self.melted & self.field.lattice) / np.sum(self.field.lattice)
            
            # 支配的な相
            dominant_phase = max(phase_counts, key=phase_counts.get).value
        else:
            delta_max = 0
            delta_mean = 0
            phase_counts = {p: 0 for p in DeformationPhase}
            yield_frac = 1.0
            fail_frac = 1.0
            dominant_phase = "FAILURE"
        
        return {
            'sigma_app': sigma_app,
            'delta_thermal': delta_thermal,
            'delta_mech_max': np.max(delta_mech[mask]) if np.any(mask) else 0,
            'delta_max': delta_max,
            'delta_mean': delta_mean,
            'yield_frac': yield_frac,
            'fail_frac': fail_frac,
            'n_failed': np.sum(self.melted & self.field.lattice),
            'phase_counts': phase_counts,
            'dominant_phase': dominant_phase,
            'delta_distribution': delta_total[active_mask].copy() if n_active > 0 else np.array([]),
        }
    
    def _cascade_Z_reduction(self, newly_failed: np.ndarray):
        """カスケード効果: 破壊点の隣接原子のZを減少"""
        N = self.N
        
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if not newly_failed[i, j, k]:
                        continue
                    
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i+di, j+dj, k+dk
                                if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                                    if self.field.lattice[ni, nj, nk] and not self.melted[ni, nj, nk]:
                                        self.field.Z[ni, nj, nk] = max(1, self.field.Z[ni, nj, nk] - 1)
    
    def plot_results(self, save_path: str = None):
        """結果を可視化"""
        if not self.history:
            print("No results to plot!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # データ抽出
        sigmas = [r['sigma_app']/1e6 for r in self.history]
        delta_maxs = [r['delta_max'] for r in self.history]
        delta_means = [r['delta_mean'] for r in self.history]
        yield_fracs = [r['yield_frac']*100 for r in self.history]
        fail_fracs = [r['fail_frac']*100 for r in self.history]
        
        # 1. δ vs σ
        ax1 = axes[0, 0]
        ax1.plot(sigmas, delta_maxs, 'ro-', lw=2, label='δ_max')
        ax1.plot(sigmas, delta_means, 'bo-', lw=2, label='δ_mean')
        ax1.axhline(self.material.delta_L, color='black', linestyle='--', 
                   lw=2, label=f'δ_L = {self.material.delta_L}')
        ax1.axhline(self.engine.DELTA_YIELD, color='orange', linestyle='--', 
                   lw=1.5, label=f'δ_yield = {self.engine.DELTA_YIELD}')
        ax1.set_xlabel('Applied Stress (MPa)')
        ax1.set_ylabel('Lindemann ratio δ')
        ax1.set_title('δ Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Yield/Fail fraction
        ax2 = axes[0, 1]
        ax2.plot(sigmas, yield_fracs, 'yo-', lw=2, label='Yield (δ > 0.05)')
        ax2.plot(sigmas, fail_fracs, 'ro-', lw=2, label='Fail (δ > δ_L)')
        ax2.set_xlabel('Applied Stress (MPa)')
        ax2.set_ylabel('Fraction (%)')
        ax2.set_title('Yield & Failure Progression')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. δ分布（最終ステップ）
        ax3 = axes[0, 2]
        final = self.history[-1]
        if len(final['delta_distribution']) > 0:
            bins = np.linspace(0, 0.15, 50)
            ax3.hist(final['delta_distribution'], bins=bins, alpha=0.7, 
                    color='skyblue', edgecolor='black', label='All sites')
            
            # テール部分をハイライト
            tail = final['delta_distribution'][final['delta_distribution'] > self.material.delta_L]
            if len(tail) > 0:
                ax3.hist(tail, bins=bins, alpha=0.9, color='red',
                        edgecolor='darkred', label=f'δ > δ_L ({100*len(tail)/len(final["delta_distribution"]):.2f}%)')
            
            ax3.axvline(self.material.delta_L, color='black', linestyle='--', 
                       lw=2, label=f'δ_L = {self.material.delta_L}')
        
        ax3.set_xlabel('δ')
        ax3.set_ylabel('Count')
        ax3.set_title(f'δ Distribution (σ = {final["sigma_app"]/1e6:.1f} MPa)')
        ax3.legend()
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. 応力集中分布
        ax4 = axes[1, 0]
        K_t = self.field.stress_concentration_factor()
        K_t_active = K_t[self.field.lattice]
        ax4.hist(K_t_active, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Stress Concentration Factor K_t')
        ax4.set_ylabel('Count')
        ax4.set_title('K_t Distribution (from Z_eff)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Z分布
        ax5 = axes[1, 1]
        Z_active = self.field.Z[self.field.lattice & ~self.melted]
        if len(Z_active) > 0:
            ax5.hist(Z_active, bins=range(0, self.material.Z_bulk+3), alpha=0.7, 
                    color='green', edgecolor='black', align='left')
        ax5.set_xlabel('Coordination Number Z')
        ax5.set_ylabel('Count')
        ax5.set_title('Z Distribution (active sites)')
        ax5.grid(True, alpha=0.3)
        
        # 6. 相分布（最終）
        ax6 = axes[1, 2]
        if final['phase_counts']:
            phases = [p.value for p in DeformationPhase]
            counts = [final['phase_counts'].get(p, 0) for p in DeformationPhase]
            colors = ['blue', 'green', 'yellow', 'orange', 'red']
            bars = ax6.bar(phases, counts, color=colors, edgecolor='black', alpha=0.7)
            
            # パーセント表示
            total = sum(counts)
            if total > 0:
                for bar, count in zip(bars, counts):
                    pct = count / total * 100
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax6.set_xlabel('Deformation Phase')
        ax6.set_ylabel('Count')
        ax6.set_title('Phase Distribution (final)')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Unified δ-Theory Tensile Test: {self.material.name} @ {self.history[0].get("T", 300)}K',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved: {save_path}")
        
        return fig


# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("UNIFIED δ-THEORY TENSILE TEST")
    print("="*80)
    
    # 銅でテスト
    sim = TensileTestV3(
        N=50,
        material=MaterialData.FCC_Cu(),
        vacancy_fraction=0.02
    )
    
    # 引張試験（降伏応力まで）
    results = sim.run_test(
        sigma_max=sim.material.sigma_y * 1.5,  # 降伏応力の1.5倍まで
        n_steps=30,
        T=300.0
    )
    
    # 可視化
    fig = sim.plot_results('/content/tensile_v3_Cu_300K.png')
    
    # サマリ
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    final = results[-1]
    print(f"Material: {sim.material.name}")
    print(f"Final σ: {final['sigma_app']/1e6:.1f} MPa")
    print(f"δ_max: {final['delta_max']:.4f}")
    print(f"δ_mean: {final['delta_mean']:.4f}")
    print(f"Yield fraction: {final['yield_frac']*100:.2f}%")
    print(f"Fail fraction: {final['fail_frac']*100:.2f}%")
    print(f"Dominant phase: {final['dominant_phase']}")
    
    # テール分析
    if len(final['delta_distribution']) > 0:
        tail_frac = np.sum(final['delta_distribution'] > sim.material.delta_L) / len(final['delta_distribution'])
        print(f"\nTail analysis:")
        print(f"  Sites with δ > δ_L: {tail_frac*100:.3f}%")
        print(f"  → These are dislocation sources!")
    
    # 高温テスト
    print("\n" + "="*80)
    print("HIGH TEMPERATURE TEST (T = 800K)")
    print("="*80)
    
    sim2 = TensileTestV3(
        N=50,
        material=MaterialData.FCC_Cu(),
        vacancy_fraction=0.02
    )
    
    results2 = sim2.run_test(
        sigma_max=sim2.material.sigma_y,
        n_steps=30,
        T=800.0
    )
    
    fig2 = sim2.plot_results('/content/tensile_v3_Cu_800K.png')
    
    plt.show()
