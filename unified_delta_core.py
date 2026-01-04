#!/usr/bin/env python3
"""
Unified δ-Theory Core Engine
=============================

統一δ理論の核心計算ロジック

用途:
  - 引張試験シミュレーション
  - プレス成形解析
  - クリープ予測
  - 疲労評価

核心原理:
  δ_total = δ_thermal + δ_mech
  
  δ_thermal = √(kT/Mω²) / r_nn     # 熱揺らぎ（温度依存）
  δ_mech = σ_local / E(T)          # 弾性ひずみのみ！
  
  σ_local = σ_nominal × K_t        # 応力集中
  
  K_t sources:
    - 空孔: K_t = 1 + A/√r
    - 曲げ: K_t = 1 + t/(2R)
    - 配位数: K_t = Z_bulk / Z_eff
    - 板厚減少: K_t = t_0 / t

相図:
  δ < 0.01: Hooke（完全弾性）
  δ < 0.03: 非線形弾性
  δ < 0.05: 降伏域（転位活性化）
  δ < δ_L:  塑性流動
  δ ≥ δ_L:  Lindemann（破壊/融解）

Author: Tamaki & Masamichi
Date: 2025-01-04
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum

# 物理定数
k_B = 1.380649e-23  # Boltzmann定数 [J/K]
u_kg = 1.66053906660e-27  # 原子質量単位 [kg]


class DeformationPhase(Enum):
    """変形相（統一δ理論）"""
    HOOKE = "HOOKE"              # δ < 0.01
    NONLINEAR = "NONLINEAR"      # δ < 0.03
    YIELD = "YIELD"              # δ < 0.05
    PLASTIC = "PLASTIC"          # δ < δ_L
    FAILURE = "FAILURE"          # δ ≥ δ_L


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
            name="FCC-Cu", structure="FCC", Z_bulk=12,
            a_300K=3.61e-10, alpha=1.7e-5,
            E0=130e9, nu=0.34, T_melt=1357,
            M_amu=63.546, rho=8960,
            delta_L=0.10, sigma_y=122e6,
            lambda_base=26.3, kappa=1.713,
        )
    
    @classmethod
    def FCC_Al(cls):
        """FCCアルミニウム"""
        return cls(
            name="FCC-Al", structure="FCC", Z_bulk=12,
            a_300K=4.05e-10, alpha=2.3e-5,
            E0=70e9, nu=0.33, T_melt=933,
            M_amu=26.982, rho=2700,
            delta_L=0.11, sigma_y=35e6,
            lambda_base=27.3, kappa=4.180,
        )
    
    @classmethod
    def BCC_Fe(cls):
        """BCC鉄"""
        return cls(
            name="BCC-Fe", structure="BCC", Z_bulk=8,
            a_300K=2.87e-10, alpha=1.5e-5,
            E0=210e9, nu=0.29, T_melt=1811,
            M_amu=55.845, rho=7870,
            delta_L=0.18, sigma_y=250e6,
            lambda_base=49.2, kappa=0.573,
        )
    
    @classmethod
    def SECD(cls):
        """SECD（電気亜鉛めっき鋼板）≈ BCC-Fe"""
        return cls(
            name="SECD", structure="BCC", Z_bulk=8,
            a_300K=2.87e-10, alpha=1.5e-5,
            E0=210e9, nu=0.29, T_melt=1811,
            M_amu=55.845, rho=7870,
            delta_L=0.18, sigma_y=160e6,  # SECDは軟鋼
            lambda_base=49.2, kappa=0.573,
        )


class UnifiedDeltaEngine:
    """
    統一δ理論エンジン
    
    Usage:
        engine = UnifiedDeltaEngine(MaterialData.SECD())
        
        # 熱的δ
        delta_th = engine.delta_thermal(T=300)
        
        # 機械的δ
        delta_mech = engine.delta_mechanical(sigma_local=200e6, T=300)
        
        # 合計
        delta_total = delta_th + delta_mech
        
        # 相判定
        phase = engine.determine_phase(delta_total)
    """
    
    # 相境界（δ値）
    DELTA_HOOKE = 0.01
    DELTA_NONLINEAR = 0.03
    DELTA_YIELD = 0.05
    
    def __init__(self, material: MaterialData):
        self.mat = material
        self.M = material.M_amu * u_kg
        
        # 室温弾性定数
        self.G0 = material.E0 / (2.0 * (1.0 + material.nu))
        self.K0 = material.E0 / (3.0 * (1.0 - 2.0 * material.nu))
    
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
    
    def bulk_modulus(self, T: float) -> float:
        """温度依存体積弾性率 K(T) [Pa]"""
        return self.K0 * (1.0 - 0.3 * (T / self.mat.T_melt)**2)
    
    # ========================================
    # δ計算（統一理論）
    # ========================================
    
    def delta_thermal(self, T: float) -> float:
        """
        熱的δ成分
        
        δ_thermal = √⟨u²⟩ / r_nn
        """
        if T <= 0:
            return 0.0
        
        G = self.shear_modulus(T)
        K = self.bulk_modulus(T)
        a = self.lattice_constant(T)
        
        # 数密度
        if self.mat.structure == "FCC":
            n = 4.0 / (a**3)
        elif self.mat.structure == "BCC":
            n = 2.0 / (a**3)
        else:
            n = 4.0 / (a**3)
        
        rho = n * self.M
        
        # 音速
        v_t = math.sqrt(max(G / rho, 1.0))
        v_l = math.sqrt(max((K + 4.0*G/3.0) / rho, 1.0))
        
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
        """
        E_T = self.youngs_modulus(T)
        if E_T <= 0:
            return 0.0
        return abs(sigma_local) / E_T
    
    def delta_total(self, sigma_local: float, T: float) -> float:
        """合計δ = δ_thermal + δ_mech"""
        return self.delta_thermal(T) + self.delta_mechanical(sigma_local, T)
    
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
        elif delta < self.mat.delta_L:
            return DeformationPhase.PLASTIC
        else:
            return DeformationPhase.FAILURE
    
    def is_yielded(self, delta: float) -> bool:
        """降伏したか（δ > δ_yield）"""
        return delta >= self.DELTA_YIELD
    
    def is_failed(self, delta: float) -> bool:
        """破壊したか（δ ≥ δ_L）"""
        return delta >= self.mat.delta_L


class StressConcentrationCalculator:
    """
    応力集中係数 K_t の計算
    
    プレス成形での応力集中源:
      1. 曲げR: K_t = 1 + t/(2R)
      2. 配位数: K_t = Z_bulk / Z_eff
      3. 板厚減少: K_t = t_0 / t
      4. 複合: K_t = K_t_bend × K_t_Z × K_t_thin
    """
    
    @staticmethod
    def from_bending_radius(thickness: float, R: float, R_min: float = 0.1) -> float:
        """
        曲げRからの応力集中
        
        K_t = 1 + t / (2R)
        
        Args:
            thickness: 板厚 [mm]
            R: 曲げ半径 [mm]
            R_min: 最小R（発散防止）[mm]
        
        Returns:
            K_t
        """
        R_eff = max(R, R_min)
        return 1.0 + thickness / (2.0 * R_eff)
    
    @staticmethod
    def from_coordination(Z_bulk: int, Z_eff: float) -> float:
        """
        配位数からの応力集中
        
        K_t = Z_bulk / Z_eff
        
        低Z = 結合少ない = 応力集中
        """
        return Z_bulk / max(Z_eff, 0.5)
    
    @staticmethod
    def from_thinning(t_0: float, t_current: float) -> float:
        """
        板厚減少からの応力集中
        
        K_t = t_0 / t_current
        """
        return t_0 / max(t_current, t_0 * 0.1)
    
    @staticmethod
    def from_curvature(curvature: float, thickness: float) -> float:
        """
        曲率からの応力集中
        
        K_t = 1 + t × |κ| / 2
        
        Args:
            curvature: 曲率 [1/mm]
            thickness: 板厚 [mm]
        """
        return 1.0 + thickness * abs(curvature) / 2.0
    
    @staticmethod
    def combined(K_t_list: list) -> float:
        """
        複合応力集中
        
        K_t_total = max(K_t) × (1 + 0.1 × min(K_t))
        
        最大のK_tが支配的、他は補正
        """
        if not K_t_list:
            return 1.0
        
        K_max = max(K_t_list)
        K_min = min(K_t_list)
        
        return K_max * (1.0 + 0.1 * K_min)


class CascadeEngine:
    """
    カスケード崩壊エンジン
    
    崩壊 → 発熱 → δ_thermal増加 → さらに崩壊
    
    これが白層/シアバンド形成の物理的メカニズム！
    
    重要な物理：
      T_melt_local = T_melt_bulk × (Z_eff / Z_bulk)^α
      
      せん断帯境界でZが低下 → 融点が下がる
      → より低い温度で「溶ける」
      → 白層形成しやすい！
    """
    
    # 結合エネルギー [eV]
    BOND_ENERGY = {
        'FCC-Cu': 3.49,
        'FCC-Al': 3.39,
        'BCC-Fe': 4.28,
        'SECD': 4.28,
    }
    
    eV_to_J = 1.602176e-19
    
    # Z依存融点のスケーリング指数
    # T_melt(Z) = T_melt_bulk × (Z/Z_bulk)^α
    # α ≈ 1.0-1.5 (Lindemann + Gibbs-Thomson から)
    ALPHA_MELT = 1.2
    
    def __init__(self, material: MaterialData, efficiency: float = 0.1):
        """
        Args:
            material: 材料データ
            efficiency: 熱変換効率（0.1 = 10%が熱に）
        """
        self.mat = material
        self.engine = UnifiedDeltaEngine(material)
        self.efficiency = efficiency
        
        # 結合エネルギー
        self.E_bond = self.BOND_ENERGY.get(material.name, 4.0) * self.eV_to_J
        
        # 1結合切断あたりの温度上昇
        # ΔT = E_bond / (3 k_B) × efficiency
        self.dT_per_bond = self.E_bond / (3 * k_B) * efficiency
        
        print(f"CascadeEngine: {material.name}")
        print(f"  E_bond = {self.E_bond/self.eV_to_J:.2f} eV")
        print(f"  ΔT per bond = {self.dT_per_bond:.1f} K (η={efficiency})")
        print(f"  T_melt scaling: T_m(Z) = {material.T_melt}K × (Z/{material.Z_bulk})^{self.ALPHA_MELT}")
    
    def local_melting_temperature(self, Z_eff: np.ndarray) -> np.ndarray:
        """
        Z依存の局所融点を計算
        
        T_melt_local = T_melt_bulk × (Z_eff / Z_bulk)^α
        
        物理：
          - バルク (Z=8): T_melt = 1811 K
          - 表面 (Z=4):   T_melt ≈ 1811 × 0.5^1.2 ≈ 790 K
          - エッジ (Z=3): T_melt ≈ 1811 × 0.375^1.2 ≈ 540 K
          
        これがGibbs-Thomson効果の一般化！
        """
        Z_ratio = np.clip(Z_eff / self.mat.Z_bulk, 0.1, 1.0)
        return self.mat.T_melt * (Z_ratio ** self.ALPHA_MELT)
    
    def is_locally_molten(self, T_local: np.ndarray, Z_eff: np.ndarray) -> np.ndarray:
        """
        局所的に融解しているか判定
        
        T > T_melt_local(Z) なら融解！
        """
        T_melt_local = self.local_melting_temperature(Z_eff)
        return T_local > T_melt_local
    
    def cascade_step(self, 
                     delta_total: np.ndarray,
                     T_local: np.ndarray,
                     Z_eff: np.ndarray,
                     neighbors: list,
                     sigma_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        1回のカスケードステップ
        
        1. δ ≥ δ_L の原子を崩壊
        2. 結合切断 → 発熱 → 隣接原子のT上昇
        3. T上昇 → δ_thermal増加 → δ_total更新
        4. 隣接原子のZ低下 → K_t増加 → δ_mech増加
        
        Returns:
            delta_total: 更新後のδ
            T_local: 更新後の温度
            Z_eff: 更新後の配位数
            n_new_collapsed: 新しく崩壊した数
        """
        # 現在の崩壊マスク
        collapsed = delta_total >= self.mat.delta_L
        n_collapsed_before = np.sum(collapsed)
        
        if n_collapsed_before == 0:
            return delta_total, T_local, Z_eff, 0
        
        # 崩壊した原子の処理
        for i in np.where(collapsed)[0]:
            # この原子の近傍
            nb = neighbors[i] if i < len(neighbors) else []
            
            for j in nb:
                if j >= len(T_local):
                    continue
                
                # 1. 発熱！（超重要）
                T_local[j] += self.dT_per_bond
                
                # 2. Z低下（結合が切れた）
                Z_eff[j] = max(Z_eff[j] - 1, 0.5)
        
        # δを再計算
        # δ_thermal（温度依存）
        delta_thermal_new = np.array([self.engine.delta_thermal(min(T, self.mat.T_melt * 2)) for T in T_local])
        
        # δ_mech（Z低下による応力集中）
        K_t_new = self.mat.Z_bulk / np.maximum(Z_eff, 0.5)
        E_T = np.array([max(self.engine.youngs_modulus(min(T, self.mat.T_melt * 2)), 1e6) for T in T_local])
        delta_mech_new = (sigma_local * K_t_new) / E_T
        
        # 無限大/NaN防止
        delta_mech_new = np.clip(delta_mech_new, 0, 10.0)
        
        # 合計
        delta_total_new = delta_thermal_new + delta_mech_new
        
        # 新しく崩壊した数
        collapsed_new = delta_total_new >= self.mat.delta_L
        n_new = np.sum(collapsed_new) - n_collapsed_before
        
        return delta_total_new, T_local, Z_eff, max(n_new, 0)
    
    def run_cascade(self,
                    delta_total: np.ndarray,
                    T_local: np.ndarray,
                    Z_eff: np.ndarray,
                    neighbors: list,
                    sigma_local: np.ndarray,
                    max_iterations: int = 20) -> Dict:
        """
        カスケードを収束まで実行
        
        Returns:
            dict with final state and statistics
        """
        delta = delta_total.copy()
        T = T_local.copy()
        Z = Z_eff.copy()
        
        total_collapsed = np.sum(delta >= self.mat.delta_L)
        cascade_history = [total_collapsed]
        
        for iteration in range(max_iterations):
            delta, T, Z, n_new = self.cascade_step(delta, T, Z, neighbors, sigma_local)
            
            total_collapsed = np.sum(delta >= self.mat.delta_L)
            cascade_history.append(total_collapsed)
            
            if n_new == 0:
                break
        
        return {
            'delta_total': delta,
            'T_local': T,
            'Z_eff': Z,
            'cascade_history': cascade_history,
            'iterations': iteration + 1,
            'final_collapsed': total_collapsed,
            'T_max': T.max(),
            # Z依存融点で白層判定！
            'T_melt_local': self.local_melting_temperature(Z),
            'locally_molten': self.is_locally_molten(T, Z),
            'white_layer_fraction': np.mean(self.is_locally_molten(T, Z)),
            'is_shear_band': np.any(self.is_locally_molten(T, Z)),  # 1箇所でも融解したら
        }


def compute_delta_map(
    vertices: np.ndarray,
    curvatures: np.ndarray,
    Z_eff: np.ndarray,
    thickness: float,
    sigma_nominal: float,
    T: float,
    material: MaterialData
) -> Dict:
    """
    CAD形状からδマップを計算（プレス成形用）
    
    Args:
        vertices: 頂点座標 [N, 3] (mm)
        curvatures: 各頂点の曲率 [N] (1/mm)
        Z_eff: 有効配位数 [N]
        thickness: 板厚 (mm)
        sigma_nominal: 公称応力 (Pa)
        T: 温度 (K)
        material: 材料データ
    
    Returns:
        dict with:
          - delta_thermal: 熱的δ（スカラー）
          - delta_mech: 機械的δ [N]
          - delta_total: 合計δ [N]
          - K_t: 応力集中係数 [N]
          - phases: 変形相 [N]
          - fail_mask: 破壊フラグ [N]
    """
    engine = UnifiedDeltaEngine(material)
    calc = StressConcentrationCalculator()
    
    N = len(vertices)
    
    # 1. 熱的δ（全点共通）
    delta_thermal = engine.delta_thermal(T)
    
    # 2. 応力集中
    K_t = np.ones(N)
    for i in range(N):
        K_t_curv = calc.from_curvature(curvatures[i], thickness)
        K_t_Z = calc.from_coordination(material.Z_bulk, Z_eff[i])
        K_t[i] = calc.combined([K_t_curv, K_t_Z])
    
    # 3. 局所応力
    sigma_local = K_t * sigma_nominal
    
    # 4. 機械的δ
    E_T = engine.youngs_modulus(T)
    delta_mech = sigma_local / E_T
    
    # 5. 合計δ
    delta_total = delta_thermal + delta_mech
    
    # 6. 相判定
    phases = np.array([engine.determine_phase(d) for d in delta_total])
    fail_mask = delta_total >= material.delta_L
    
    return {
        'delta_thermal': delta_thermal,
        'delta_mech': delta_mech,
        'delta_total': delta_total,
        'K_t': K_t,
        'sigma_local': sigma_local,
        'phases': phases,
        'fail_mask': fail_mask,
        'fail_fraction': np.sum(fail_mask) / N,
        'yield_fraction': np.sum(delta_total >= engine.DELTA_YIELD) / N,
    }


# ========================================
# テスト
# ========================================
if __name__ == "__main__":
    print("="*60)
    print("Unified δ-Theory Core Engine Test")
    print("="*60)
    
    # 各材料でテスト
    materials = [
        MaterialData.FCC_Cu(),
        MaterialData.FCC_Al(),
        MaterialData.BCC_Fe(),
        MaterialData.SECD(),
    ]
    
    print(f"\n{'Material':<12} {'δ_L':<8} {'δ_th(300K)':<12} {'δ_th(Tm/2)':<12} {'E(300K)':<12}")
    print("-"*60)
    
    for mat in materials:
        engine = UnifiedDeltaEngine(mat)
        
        delta_300 = engine.delta_thermal(300)
        delta_half = engine.delta_thermal(mat.T_melt / 2)
        E_300 = engine.youngs_modulus(300) / 1e9
        
        print(f"{mat.name:<12} {mat.delta_L:<8.2f} {delta_300:<12.4f} {delta_half:<12.4f} {E_300:<12.1f}")
    
    # SECDでプレス成形テスト
    print("\n" + "="*60)
    print("SECD Press Forming Test")
    print("="*60)
    
    secd = MaterialData.SECD()
    engine = UnifiedDeltaEngine(secd)
    
    # 仮想CADデータ
    N = 1000
    np.random.seed(42)
    
    vertices = np.random.randn(N, 3) * 50  # mm
    curvatures = np.abs(np.random.randn(N) * 0.1)  # 1/mm
    Z_eff = np.random.randint(4, 9, N).astype(float)  # 配位数
    
    # テスト条件
    thickness = 1.96  # mm
    sigma_nominal = 200e6  # Pa (200 MPa)
    T = 350  # K (プレス加工中の温度上昇)
    
    result = compute_delta_map(
        vertices, curvatures, Z_eff,
        thickness, sigma_nominal, T, secd
    )
    
    print(f"\nConditions:")
    print(f"  t = {thickness} mm")
    print(f"  σ_nominal = {sigma_nominal/1e6:.0f} MPa")
    print(f"  T = {T} K")
    
    print(f"\nResults:")
    print(f"  δ_thermal = {result['delta_thermal']:.4f}")
    print(f"  δ_mech: [{result['delta_mech'].min():.4f}, {result['delta_mech'].max():.4f}]")
    print(f"  δ_total: [{result['delta_total'].min():.4f}, {result['delta_total'].max():.4f}]")
    print(f"  K_t: [{result['K_t'].min():.2f}, {result['K_t'].max():.2f}]")
    print(f"  Yield fraction: {result['yield_fraction']*100:.1f}%")
    print(f"  Fail fraction: {result['fail_fraction']*100:.2f}%")
    
    # 相分布
    print(f"\nPhase distribution:")
    for phase in DeformationPhase:
        count = np.sum(result['phases'] == phase)
        print(f"  {phase.value}: {count} ({count/N*100:.1f}%)")
    
    # CascadeEngineテスト
    print("\n" + "="*60)
    print("Cascade Engine Test (White Layer / Shear Band)")
    print("="*60)
    
    cascade = CascadeEngine(secd, efficiency=0.1)
    
    # Z依存融点のデモ
    print("\n--- Z-dependent melting temperature ---")
    Z_demo = np.array([8, 6, 4, 3, 2])
    T_melt_demo = cascade.local_melting_temperature(Z_demo)
    print(f"  {'Z':<6} {'T_melt [K]':<12} {'vs bulk':<10}")
    for z, tm in zip(Z_demo, T_melt_demo):
        ratio = tm / secd.T_melt
        print(f"  {z:<6} {tm:<12.0f} {ratio*100:.0f}%")
    
    print("\n  → 低Zほど融点が下がる！")
    print("  → せん断帯境界(Z≈3-4)では T_melt ≈ 500-800 K")
    
    # 極端な条件でカスケードをトリガー
    N_test = 100
    
    # 初期条件（かなり極端に）
    T_local = np.full(N_test, 800.0)  # 800K（高温プレス）
    Z_test = np.random.randint(2, 5, N_test).astype(float)  # 非常に低配位数
    sigma_test = np.full(N_test, 2000e6)  # 2000 MPa（超高応力）
    
    # 近傍リスト（線形チェーン）
    neighbors_test = []
    for i in range(N_test):
        nb = []
        if i > 0: nb.append(i-1)
        if i < N_test - 1: nb.append(i+1)
        neighbors_test.append(nb)
    
    # 初期δ計算
    delta_thermal_init = cascade.engine.delta_thermal(800)
    K_t_init = secd.Z_bulk / np.maximum(Z_test, 0.5)
    E_init = cascade.engine.youngs_modulus(800)
    delta_mech_init = sigma_test * K_t_init / E_init
    delta_init = delta_thermal_init + delta_mech_init
    
    print(f"\nInitial state (EXTREME conditions):")
    print(f"  T = 800 K (high temp forming)")
    print(f"  σ = 2000 MPa (extreme stress)")
    print(f"  Z range: [{Z_test.min():.0f}, {Z_test.max():.0f}] (defect rich)")
    print(f"  E(800K) = {E_init/1e9:.1f} GPa")
    print(f"  δ_thermal = {delta_thermal_init:.4f}")
    print(f"  δ_mech range: [{delta_mech_init.min():.4f}, {delta_mech_init.max():.4f}]")
    print(f"  δ_total range: [{delta_init.min():.4f}, {delta_init.max():.4f}]")
    print(f"  δ_L = {secd.delta_L}")
    print(f"  Initially collapsed: {np.sum(delta_init >= secd.delta_L)}")
    
    # カスケード実行
    cascade_result = cascade.run_cascade(
        delta_init, T_local.copy(), Z_test.copy(),
        neighbors_test, sigma_test, max_iterations=20
    )
    
    print(f"\nCascade result:")
    print(f"  Iterations: {cascade_result['iterations']}")
    print(f"  Final collapsed: {cascade_result['final_collapsed']} / {N_test}")
    print(f"  T_max: {cascade_result['T_max']:.1f} K")
    print(f"  White layer fraction: {cascade_result['white_layer_fraction']*100:.1f}%")
    print(f"  Is shear band: {cascade_result['is_shear_band']}")
    
    # カスケード履歴
    hist = cascade_result['cascade_history']
    if len(hist) > 1 and hist[-1] > hist[0]:
        print(f"\n  🔥 CASCADE OCCURRED!")
        print(f"     {hist[0]} → {hist[-1]} collapsed")
        print(f"     Amplification: {hist[-1]/max(hist[0],1):.1f}x")
    
    # 手動で一部を崩壊状態に
    print("\n--- Manual trigger test (single seed) ---")
    delta_manual = delta_init.copy()
    delta_manual[50] = 0.20  # 中央を崩壊状態に
    
    cascade_result2 = cascade.run_cascade(
        delta_manual, T_local.copy(), Z_test.copy(),
        neighbors_test, sigma_test, max_iterations=20
    )
    
    print(f"  Seed: 1 collapsed site at center")
    print(f"  Final collapsed: {cascade_result2['final_collapsed']}")
    print(f"  T_max: {cascade_result2['T_max']:.1f} K")
    print(f"  T_melt_local (at cascade sites):")
    
    # カスケードで影響を受けたサイトの詳細
    T_final = cascade_result2['T_local']
    Z_final = cascade_result2['Z_eff']
    T_melt_local = cascade_result2['T_melt_local']
    molten = cascade_result2['locally_molten']
    
    molten_idx = np.where(molten)[0]
    if len(molten_idx) > 0:
        print(f"    Molten sites: {len(molten_idx)}")
        # サンプル表示
        for idx in molten_idx[:5]:
            print(f"      Site {idx}: T={T_final[idx]:.0f}K > T_melt={T_melt_local[idx]:.0f}K (Z={Z_final[idx]:.1f})")
        if len(molten_idx) > 5:
            print(f"      ... and {len(molten_idx)-5} more")
    
    print(f"\n  White layer fraction: {cascade_result2['white_layer_fraction']*100:.1f}%")
    
    if cascade_result2['final_collapsed'] > 1:
        print(f"\n  🔥 CASCADE + WHITE LAYER from single seed!")
        print(f"     This is how shear bands form in reality!")
