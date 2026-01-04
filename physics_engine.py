"""
Λ³-Dynamics Physics Engine
===========================

統一物理エンジン - 全ての計算ロジックを1箇所に

Components:
  1. Geometry          - 格子定数、原子間距離、数密度
  2. Thermal Softening - 熱膨張による連続的な剛性低下
  3. Born Collapse     - 構造依存の臨界崩壊係数（Z³スケーリング）
  4. Debye-Waller      - 熱的原子振動（U²_thermal）
  5. Thermal Fluctuation - 確率的閾値超え
  6. Hooke             - 機械的変形（U²_mech）
  7. Lindemann         - 臨界判定（U²_c、λ）

重要な区別:
  ┌────────────────────────────────────────────────────────────┐
  │ Thermal Softening (連続的)                                  │
  │   E(T)/E₀ = exp[-λ_eff × α × ΔT]                           │
  │   格子が熱膨張で広がる → 結合弱化 → 剛性↓                   │
  │   → U²_thermal の計算に反映（G(T)経由）                     │
  ├────────────────────────────────────────────────────────────┤
  │ Born Collapse (臨界的)                                      │
  │   f_G = f_G⁰ × (Z_eff/12)³                                  │
  │   格子が「壊れる」瞬間の構造依存係数                         │
  │   → U²_c の計算に反映（Z³スケーリング）                     │
  └────────────────────────────────────────────────────────────┘

使用法:
  from physics_engine import PhysicsEngine, create_engine
  
  engine = create_engine('Cu')
  
  # 熱軟化（連続的）
  soft = engine.thermal_softening(1000)  # E(1000K)/E₀
  
  # Born崩壊係数（構造依存）
  fG = engine.born_collapse_fG(Z_eff=8)  # BCC的な配位
  
  # 統合判定
  lam = engine.compute_lambda(strain, T=500, Z_eff=Z_array)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from materials import PHYSICAL_CONSTANTS, get_material

# 物理定数
k_B = PHYSICAL_CONSTANTS['k_B']
u_kg = PHYSICAL_CONSTANTS['amu']


# ============================================================
# PhysicsEngine
# ============================================================
class PhysicsEngine:
    """
    Λ³統一物理エンジン
    
    Core equations:
      Λ = K / |V|_eff = U² / U²_c
      
    二つの独立した効果:
      1. Thermal Softening: G(T) = G₀ × exp[-λ_eff α ΔT]
         → 連続的な剛性低下 → U²_thermal 増加
         
      2. Born Collapse: f_G = 0.097 × (Z_eff/12)³
         → 構造依存の臨界閾値 → U²_c に反映
    
    λ判定:
      λ = U²_total / U²_c
      λ < 1: 安定
      λ ≥ 1: 臨界（融解/破壊）
      
    熱揺らぎ:
      P_exceed = exp(-(1-λ)/σ_λ)
      高温ではσ_λ大 → 確率的に閾値超え可能
    """
    
    # Z³スケーリング定数（7金属検証済み）
    FG_FCC_REF = 0.097   # Reference: FCC at Z=12
    Z_FCC_REF = 12       # Reference coordination
    
    # 熱揺らぎの物理定数
    NU_0 = 1e13  # Debye周波数（試行頻度）[Hz]
    
    def __init__(self, material: dict):
        """
        Args:
            material: 材料データ（get_material()の戻り値）
        """
        self.mat = material
        self.M = material['M_amu'] * u_kg  # kg
        
        # 室温の弾性定数
        self.G0 = material['E0'] / (2.0 * (1.0 + material['nu']))  # 剛性率
        self.K0 = material['E0'] / (3.0 * (1.0 - 2.0 * material['nu']))  # 体積弾性率
        
        # Z³ベースのfG計算
        self.Z_eff_bulk = self._compute_Z_eff_bulk()
        self.fG_at_melt = self._compute_fG_Z3()
        
        print("="*60)
        print("Λ³ Physics Engine Initialized")
        print("="*60)
        print(f"Material: {material['name']} ({material['structure']})")
        print(f"a(300K) = {material['a']*1e10:.3f} Å")
        if material.get('c_over_a', 0) > 0:
            print(f"c/a = {material['c_over_a']:.3f}")
        print(f"E₀ = {material['E0']/1e9:.1f} GPa")
        print(f"G₀ = {self.G0/1e9:.2f} GPa")
        print(f"T_melt = {material['T_melt']} K")
        print(f"δ_L = {material['delta_L']}")
        print(f"λ_base = {material['lambda_base']}, κ = {material['kappa']}")
        print(f"Z_eff = {self.Z_eff_bulk:.2f} (Z³ scaling)")
        print(f"f_G(T_melt) = {self.fG_at_melt:.4f}")
    
    def _compute_Z_eff_bulk(self) -> float:
        """
        バルクの有効配位数を計算
        
        BCC: Z_eff = 8
        FCC: Z_eff = 12
        HCP: Z_eff = 6 + 6×g(c/a)
        """
        structure = self.mat['structure']
        
        if structure == 'BCC':
            return 8.0
        elif structure == 'FCC':
            return 12.0
        elif structure == 'HCP':
            c_a_ideal = 1.633
            c_a_ratio = self.mat.get('c_over_a', c_a_ideal) / c_a_ideal
            g = 0.85 + 0.08 * c_a_ratio
            g = min(max(g, 0.80), 1.00)
            return 6.0 + 6.0 * g
        else:
            return 12.0  # デフォルト
    
    def _compute_fG_Z3(self) -> float:
        """
        融点での剛性率比 f_G = G(T_melt)/G₀
        
        優先順位:
          1. materials.pyに定義済みの fG_melt（実験値から逆算済み）
          2. Z³スケーリングで計算（フォールバック）
        """
        # 材料データに fG_melt があればそれを使う（検証済みの値）
        if 'fG_melt' in self.mat:
            return self.mat['fG_melt']
        
        # なければZ³スケーリングで計算（近似）
        return self.FG_FCC_REF * (self.Z_eff_bulk / self.Z_FCC_REF) ** 3
    
    # ========================================
    # 1. 幾何関数（温度依存）
    # ========================================
    
    def lattice_constant(self, T: float) -> float:
        """
        温度依存の格子定数 a(T)
        
        a(T) = a₀ × (1 + α × (T - T_ref))
        """
        return self.mat['a'] * (1.0 + self.mat['alpha'] * (T - 300.0))
    
    def nearest_neighbor_distance(self, T: float) -> float:
        """
        最近接原子間距離 r_nn(T)
        
        BCC: r_nn = a√3/2
        FCC: r_nn = a/√2
        HCP: r_nn = a
        """
        a = self.lattice_constant(T)
        structure = self.mat['structure']
        
        if structure == "BCC":
            return a * math.sqrt(3) / 2
        elif structure == "FCC":
            return a / math.sqrt(2)
        elif structure == "HCP":
            return a
        return a / math.sqrt(2)  # デフォルト
    
    def number_density(self, T: float) -> float:
        """
        原子数密度 n(T) [atoms/m³]
        
        BCC: 2/a³
        FCC: 4/a³
        HCP: 2/V_cell
        """
        a = self.lattice_constant(T)
        structure = self.mat['structure']
        
        if structure == "BCC":
            return 2.0 / (a**3)
        elif structure == "FCC":
            return 4.0 / (a**3)
        elif structure == "HCP":
            c = a * self.mat.get('c_over_a', 1.633)
            V_cell = (math.sqrt(3.0) / 2.0) * a**2 * c
            return 2.0 / V_cell
        return 4.0 / (a**3)
    
    # ========================================
    # 2. 熱軟化（Thermal Softening）- 連続的
    # ========================================
    
    def thermal_softening(self, T: float) -> float:
        """
        熱膨張による剛性低下（連続的な効果）
        
        E(T)/E₀ = exp[-λ_eff × α × ΔT]
        
        where:
          λ_eff = λ_base × (1 + κ × ΔT/1000)
          ΔT = T - T_ref (T_ref = 293K)
        
        物理的意味:
          - 格子が熱膨張で広がる → 結合弱化 → 剛性↓
          - 融点 T_melt で fG_at_melt に到達（Born崩壊閾値）
          - λ_base, κ は7金属フィッティングで決定済み
        """
        T_ref = 293.0
        
        if T <= T_ref:
            return 1.0
        
        delta_T = T - T_ref
        alpha = self.mat['alpha']
        lambda_base = self.mat['lambda_base']
        kappa = self.mat['kappa']
        
        # Λ³熱軟化モデル（7金属検証済み）
        lambda_eff = lambda_base * (1.0 + kappa * delta_T / 1000.0)
        softening = math.exp(-lambda_eff * alpha * delta_T)
        
        # 融点でBorn崩壊閾値に達する
        return max(softening, self.fG_at_melt)
    
    # ========================================
    # 3. Born Collapse（臨界的崩壊）- Z³スケーリング
    # ========================================
    
    def born_collapse_fG(self, Z_eff: float = None) -> float:
        """
        構造依存の臨界崩壊係数（Z³スケーリング）
        
        f_G = f_G⁰ × (Z_eff / 12)³
        
        f_G⁰ = 0.097 (FCC close-packed reference)
        
        物理的解釈:
          Born崩壊 = 剛性率がゼロになる瞬間（格子の臨界的破壊）
          Z³ = 3D剛性ネットワークの剛性浸透（rigidity percolation）
            - Z¹: 結合エネルギー密度
            - Z²: 角度拘束（せん断抵抗）
            - Z³: 協同崩壊確率
        
        Returns:
            f_G: 臨界崩壊係数（0～1、小さいほど壊れやすい）
        """
        if Z_eff is None:
            Z_eff = self.Z_eff_bulk
        
        return self.FG_FCC_REF * (Z_eff / self.Z_FCC_REF) ** 3
    
    # ========================================
    # 4. 弾性定数（熱軟化適用）
    # ========================================
    
    def shear_modulus(self, T: float) -> float:
        """
        温度依存の剛性率 G(T)
        
        2つのレジーム:
          T < 0.9 T_m:  Λ³ continuous softening (exponential)
          T ≥ 0.9 T_m:  Born collapse (急降下 → fG_melt)
        
        図式:
          G/G₀
            │
          1 ├● 300K
            │ ╲
            │  ╲  Λ³ softening (exp)
            │   ╲
            │    ╲___● 0.9 Tm
            │        │ Born collapse
            │        ●─── fG_melt
          0 ├────────● Tm
            └──────────→ T
        """
        T_ref = 293.0
        T_melt = self.mat['T_melt']
        T_born = 0.9 * T_melt  # Born collapse onset
        
        if T <= T_ref:
            return self.G0
        
        if T >= T_melt:
            return self.G0 * self.fG_at_melt
        
        if T < T_born:
            # Region 1: Λ³ continuous softening (exponential)
            f_T = self.thermal_softening(T)
        else:
            # Region 2: Born collapse (急降下)
            # 0.9 T_m での値から fG_melt へ急降下
            G_at_born = self.thermal_softening(T_born)
            ratio = (T - T_born) / (T_melt - T_born)
            f_T = G_at_born - (G_at_born - self.fG_at_melt) * ratio
        
        return self.G0 * f_T
    
    def bulk_modulus(self, T: float) -> float:
        """温度依存の体積弾性率 K(T)"""
        T_ratio = min(T / self.mat['T_melt'], 1.0)
        return self.K0 * (1.0 - 0.3 * T_ratio**2)
    
    def youngs_modulus(self, T: float) -> float:
        """温度依存のヤング率 E(T) = 9KG/(3K+G)"""
        G = self.shear_modulus(T)
        K = self.bulk_modulus(T)
        return 9.0 * K * G / (3.0 * K + G)
    
    # ========================================
    # 3. Debye-Waller（熱的原子振動）
    # ========================================
    
    def sound_velocities(self, T: float) -> Tuple[float, float]:
        """音速 v_t（横波）、v_l（縦波）"""
        G = self.shear_modulus(T)
        K = self.bulk_modulus(T)
        n = self.number_density(T)
        rho = n * self.M
        
        v_t = math.sqrt(G / rho)
        v_l = math.sqrt((K + 4.0*G/3.0) / rho)
        
        return v_t, v_l
    
    def debye_wavevector(self, T: float) -> float:
        """Debye波数 k_D = (6π²n)^(1/3)"""
        n = self.number_density(T)
        return (6.0 * math.pi**2 * n) ** (1.0/3.0)
    
    def inverse_omega_squared(self, T: float) -> float:
        """⟨1/ω²⟩の計算（Debye模型）"""
        v_t, v_l = self.sound_velocities(T)
        k_D = self.debye_wavevector(T)
        return (1.0 / (3.0 * k_D**2)) * (2.0/v_t**2 + 1.0/v_l**2)
    
    def thermal_displacement_squared(self, T: float) -> float:
        """
        熱的原子変位の二乗 ⟨u²⟩_thermal
        
        ⟨u²⟩ = (k_B T / M) × ⟨1/ω²⟩
        """
        if T <= 0:
            return 0.0
        
        inv_omega2 = self.inverse_omega_squared(T)
        return (k_B * T / self.M) * inv_omega2
    
    def thermal_lindemann_ratio(self, T: float) -> float:
        """
        熱的Lindemann比 δ_thermal = √⟨u²⟩ / r_nn
        """
        u2 = self.thermal_displacement_squared(T)
        r_nn = self.nearest_neighbor_distance(T)
        return math.sqrt(u2) / r_nn
    
    # ========================================
    # 4. Thermal Fluctuation（熱揺らぎの確率論）
    # ========================================
    
    def thermal_fluctuation_sigma(self, T: float) -> float:
        """
        熱揺らぎの標準偏差 σ_δ ≈ δ_thermal
        
        高温: σ_δ大 → 確率的にΛ>1を超えやすい
        低温: σ_δ小 → 確率的にΛ>1を超えにくい
        """
        return self.thermal_lindemann_ratio(T)
    
    def lambda_fluctuation_sigma(self, lam: float, T: float) -> float:
        """
        Λの揺らぎの標準偏差 σ_Λ
        
        Λ = (δ/δ_L)² より
        σ_Λ = 2√Λ / δ_L × σ_δ
        """
        sigma_delta = self.thermal_fluctuation_sigma(T)
        delta_L = self.mat['delta_L']
        return 2.0 * math.sqrt(max(lam, 0.01)) / delta_L * sigma_delta
    
    def probability_exceed_threshold(self, lam: float, T: float, threshold: float = 1.0) -> float:
        """
        熱揺らぎにより閾値を超える確率 P_exceed
        
        P_exceed = exp(-gap / σ_Λ)  if gap > 0
                 = 1.0               if gap <= 0
        """
        gap = threshold - lam
        
        if gap <= 0:
            return 1.0
        
        sigma_lam = self.lambda_fluctuation_sigma(lam, T)
        
        if sigma_lam < 1e-10:
            return 0.0
        
        return min(math.exp(-gap / sigma_lam), 1.0)
    
    def jump_rate(self, lam: float, Z_eff: float, T: float) -> float:
        """
        熱活性化ジャンプレート [1/s]
        
        rate = ν₀ × exp(-E_barrier / kT) × P_exceed
        """
        if T <= 0:
            return 0.0
        
        P_exceed = self.probability_exceed_threshold(lam, T)
        
        if P_exceed < 1e-20:
            return 0.0
        
        gap = max(1.0 - lam, 0.01)
        E_barrier = gap * (Z_eff / 12.0) * k_B * self.mat['T_melt']
        
        kT = k_B * T
        if E_barrier / kT > 100:
            return 0.0
        
        return self.NU_0 * math.exp(-E_barrier / kT) * P_exceed
    
    # ========================================
    # 5. Hooke（機械的変形）
    # ========================================
    
    def mechanical_displacement_squared(self, 
                                         strain_tensor: np.ndarray,
                                         T: float = 300.0) -> np.ndarray:
        """
        機械的原子変位の二乗 U²_mech
        
        U²_mech = ε_max² × r_nn²
        """
        r_nn = self.nearest_neighbor_distance(T)
        
        if strain_tensor.ndim == 2:
            eigvals = np.linalg.eigvalsh(strain_tensor)
            eps_max = np.max(np.abs(eigvals))
            return (eps_max * r_nn)**2
        else:
            N = strain_tensor.shape[0]
            U2_mech = np.zeros(N)
            for i in range(N):
                eigvals = np.linalg.eigvalsh(strain_tensor[i])
                eps_max = np.max(np.abs(eigvals))
                U2_mech[i] = (eps_max * r_nn)**2
            return U2_mech
    
    def mechanical_lindemann_ratio(self, 
                                   strain_tensor: np.ndarray,
                                   T: float = 300.0) -> np.ndarray:
        """機械的Lindemann比 δ_mech = √U²_mech / r_nn"""
        U2_mech = self.mechanical_displacement_squared(strain_tensor, T)
        r_nn = self.nearest_neighbor_distance(T)
        return np.sqrt(U2_mech) / r_nn
    
    # ========================================
    # 6. Lindemann / Lambda計算
    # ========================================
    
    def total_U2(self, 
                 strain_tensor: np.ndarray,
                 T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        総U²を計算
        
        U²_total = U²_thermal + U²_mech
        """
        U2_thermal = self.thermal_displacement_squared(T)
        U2_mech = self.mechanical_displacement_squared(strain_tensor, T)
        
        if isinstance(U2_mech, np.ndarray):
            U2_thermal_arr = np.full_like(U2_mech, U2_thermal)
        else:
            U2_thermal_arr = U2_thermal
        
        return U2_thermal_arr + U2_mech, U2_thermal_arr, U2_mech
    
    def critical_U2(self, Z_eff: np.ndarray = None, T: float = None) -> np.ndarray:
        """
        臨界U²_c を計算（Born崩壊閾値）
        
        U²_c = (δ_L × r_nn)² × (Z_eff / Z_bulk)³
        
        物理的解釈:
          - δ_L: Lindemann定数（材料固有、0.1〜0.18）
          - Z³: Born collapse（構造依存の剛性浸透）
          
        Note:
          熱軟化はU²_thermal側に反映される（G(T)経由）
          こちらは「壊れる閾値」なので温度に直接依存しない
          
        Args:
            Z_eff: 有効配位数（表面/エッジで低下）
            T: 温度（r_nnの熱膨張補正用、Noneなら300K）
        """
        T_ref = T if T is not None else 300.0
        r_nn = self.nearest_neighbor_distance(T_ref)
        U2_c_bulk = (self.mat['delta_L'] * r_nn)**2
        
        if Z_eff is None:
            return U2_c_bulk
        
        Z_ratio = np.asarray(Z_eff) / self.mat['Z_bulk']
        return U2_c_bulk * (Z_ratio ** 3)
    
    def compute_lambda(self,
                       strain_tensor: np.ndarray,
                       T: float,
                       Z_eff: np.ndarray = None) -> np.ndarray:
        """λ = U²_total / U²_c を計算"""
        U2_total, _, _ = self.total_U2(strain_tensor, T)
        U2_c = self.critical_U2(Z_eff)
        return U2_total / np.maximum(U2_c, 1e-30)
    
    # ========================================
    # 7. FLC（成形限界）
    # ========================================
    
    def compute_FLC0(self, thickness_mm: float) -> float:
        """
        FLC₀（平面ひずみ限界）を計算
        """
        if 'flc_points' in self.mat:
            for beta, eps1_limit in self.mat['flc_points']:
                if abs(beta) < 0.01:
                    flc0_ref = eps1_limit
                    break
            else:
                flc0_ref = 0.35
            
            t_ref = self.mat.get('flc_t_ref', thickness_mm)
            if t_ref > 0 and thickness_mm != t_ref:
                thickness_factor = math.sqrt(thickness_mm / t_ref)
                flc0 = flc0_ref * thickness_factor
            else:
                flc0 = flc0_ref
            
            return flc0
        else:
            # Keeler-Brazier式（デフォルト）
            return (23.3 + 14.1 * thickness_mm) / 100.0
    
    def compute_FLC(self, thickness_mm: np.ndarray, minor_strain: np.ndarray) -> np.ndarray:
        """
        FLC（成形限界曲線）を計算
        
        Args:
            thickness_mm: 板厚 [mm]（スカラーまたは配列）
            minor_strain: マイナーひずみ ε₂
        
        Returns:
            major_strain_limit: メジャーひずみ限界 ε₁
        """
        minor_strain = np.asarray(minor_strain)
        thickness_mm = np.asarray(thickness_mm)
        
        if 'flc_points' in self.mat:
            # 実データから補間
            flc_points = sorted(self.mat['flc_points'], key=lambda x: x[0])
            betas = np.array([p[0] for p in flc_points])
            eps1s = np.array([p[1] for p in flc_points])
            
            # FLC₀を取得
            flc0 = self.compute_FLC0(float(np.mean(thickness_mm)))
            
            # β推定
            beta_estimated = minor_strain / np.maximum(np.abs(minor_strain) + flc0, 0.01)
            beta_estimated = np.clip(beta_estimated, betas.min(), betas.max())
            
            # FLC値を補間
            major_limit_ref = np.interp(beta_estimated, betas, eps1s)
            
            # 板厚補正
            t_ref = self.mat.get('flc_t_ref', 1.96)
            if np.isscalar(thickness_mm):
                thickness_factor = math.sqrt(thickness_mm / t_ref)
            else:
                thickness_factor = np.sqrt(np.maximum(thickness_mm, 0.1) / t_ref)
            
            return major_limit_ref * thickness_factor
        else:
            # 線形近似
            flc0 = self.compute_FLC0(float(np.mean(thickness_mm)))
            left_slope = -1.0
            right_slope = 0.55
            
            major_limit = np.where(
                minor_strain < 0,
                flc0 - left_slope * minor_strain,
                flc0 + right_slope * minor_strain
            )
            return major_limit
    
    # ========================================
    # 診断
    # ========================================
    
    def report_thermal_state(self, T: float):
        """温度状態をレポート"""
        print(f"\n--- Thermal State at T = {T} K ---")
        
        a = self.lattice_constant(T)
        r_nn = self.nearest_neighbor_distance(T)
        print(f"a(T) = {a*1e10:.4f} Å")
        print(f"r_nn(T) = {r_nn*1e10:.4f} Å")
        
        fG = self.born_collapse_factor(T)
        G = self.shear_modulus(T)
        E = self.youngs_modulus(T)
        print(f"fG(T) = {fG:.3f} (Born collapse)")
        print(f"G(T) = {G/1e9:.2f} GPa")
        print(f"E(T) = {E/1e9:.1f} GPa")
        
        u2 = self.thermal_displacement_squared(T)
        delta = self.thermal_lindemann_ratio(T)
        print(f"⟨u²⟩_thermal = {u2:.4e} m²")
        print(f"δ_thermal = {delta:.4f}")
        
        if delta >= self.mat['delta_L']:
            print(f"⚠️  δ ≥ δ_L = {self.mat['delta_L']} → UNSTABLE!")
        else:
            print(f"✓ δ < δ_L = {self.mat['delta_L']} → stable")
    
    def validate_lindemann_at_melting(self) -> float:
        """融点でLindemann判定を検証"""
        print(f"\n{'='*60}")
        print(f"Lindemann Validation at T_melt = {self.mat['T_melt']} K")
        print(f"{'='*60}")
        
        delta = self.thermal_lindemann_ratio(self.mat['T_melt'])
        
        print(f"δ_thermal(T_melt) = {delta:.4f}")
        print(f"δ_L (target) = {self.mat['delta_L']}")
        
        if 0.08 <= delta <= 0.20:
            print(f"✅ Lindemann criterion validated!")
        else:
            print(f"⚠️ δ = {delta:.3f} outside typical range [0.08, 0.20]")
        
        return delta


# ============================================================
# 便利なファクトリ関数
# ============================================================
def create_engine(material_name: str) -> PhysicsEngine:
    """
    材料名からPhysicsEngineを作成
    
    Args:
        material_name: 'Fe', 'Cu', 'SECD' など
    
    Returns:
        PhysicsEngine
    """
    return PhysicsEngine(get_material(material_name))


# ============================================================
# テスト
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Λ³ PHYSICS ENGINE TEST")
    print("="*70)
    
    # 全材料テスト
    from materials import list_materials
    
    all_materials = list_materials()['pure_metals'] + list_materials()['industrial_alloys']
    
    results = []
    
    for name in ['Fe', 'Cu', 'Al', 'Ti', 'Mg']:
        print(f"\n{'='*60}")
        engine = create_engine(name)
        delta_pred = engine.thermal_lindemann_ratio(engine.mat['T_melt'])
        delta_exp = engine.mat['delta_L']
        error = abs(delta_pred - delta_exp) / delta_exp * 100
        
        results.append({
            'name': name,
            'struct': engine.mat['structure'],
            'delta_pred': delta_pred,
            'delta_exp': delta_exp,
            'error': error,
        })
        
        print(f"\nValidation at T_melt = {engine.mat['T_melt']} K:")
        print(f"  δ_pred = {delta_pred:.4f}")
        print(f"  δ_exp  = {delta_exp:.4f}")
        print(f"  Error  = {error:.1f}%")
    
    # サマリー
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Material':<10} {'Struct':<6} {'δ_pred':<8} {'δ_exp':<8} {'Error%':<8}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<10} {r['struct']:<6} {r['delta_pred']:<8.4f} "
              f"{r['delta_exp']:<8.4f} {r['error']:<8.1f}")
    
    mean_error = np.mean([r['error'] for r in results])
    print("-"*50)
    print(f"Mean Absolute Error: {mean_error:.1f}%")
    
    # 熱揺らぎテスト
    print("\n" + "="*70)
    print("THERMAL FLUCTUATION TEST")
    print("="*70)
    
    engine = create_engine('Cu')
    
    print("\nΛ vs P_exceed (T=500K):")
    print("-"*40)
    for lam in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        P = engine.probability_exceed_threshold(lam, 500)
        print(f"  Λ={lam:.2f}: P_exceed = {P:.4f}")
