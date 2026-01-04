"""
Λ³-Dynamics Physics Core
========================

統合物理エンジン：
- Born collapse（剛性率の温度崩壊）
- Debye-Waller（熱的原子振動）
- Hooke（弾性変形）
- Lindemann criterion（臨界判定）

全ての効果を一貫したフレームワークで計算
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple

# 物理定数
k_B = 1.380649e-23  # Boltzmann定数 [J/K]
u_kg = 1.66053906660e-27  # 原子質量単位 [kg]


@dataclass
class MaterialPhysics:
    """
    材料の物理定数（完全版）
    
    Born collapse、Debye、Hookeに必要な全パラメータ
    Z³スケーリング対応
    Λ³熱軟化モデル対応（λ_base, κ）
    """
    name: str
    
    # 結晶構造
    structure: str       # "BCC" or "FCC" or "HCP"
    Z_bulk: int          # バルク配位数
    
    # 格子定数
    a_300K: float        # 300Kでの格子定数 [m]
    alpha: float         # 熱膨張係数 [1/K]
    c_over_a: float      # HCP用 c/a比（FCC/BCCは0）
    
    # 弾性定数（室温）
    E0: float            # ヤング率 [Pa]
    nu: float            # ポアソン比
    
    # 熱物性
    T_melt: float        # 融点 [K]
    M_amu: float         # 原子量 [amu]
    rho: float           # 密度 [kg/m³]
    
    # Lindemann定数
    delta_L: float       # 臨界δ
    
    # Λ³熱軟化パラメータ（7材料フィッティング済み）
    lambda_base: float = 30.0  # 基準減衰係数（デフォルト: BCC/FCC平均）
    kappa: float = 2.0         # 非調和パラメータ（デフォルト）
    
    @classmethod
    def BCC_Fe(cls):
        """BCC鉄（α-Fe）"""
        return cls(
            name="BCC-Fe",
            structure="BCC",
            Z_bulk=8,
            a_300K=2.87e-10,
            alpha=1.5e-5,  # 15e-6 from fitting data
            c_over_a=0.0,
            E0=210e9,
            nu=0.29,
            T_melt=1811,
            M_amu=55.845,
            rho=7870,
            delta_L=0.18,
            # Λ³ fitted parameters
            lambda_base=49.2,
            kappa=0.573,
        )
    
    @classmethod
    def FCC_Cu(cls):
        """FCC銅"""
        return cls(
            name="FCC-Cu",
            structure="FCC",
            Z_bulk=12,
            a_300K=3.61e-10,
            alpha=1.7e-5,
            c_over_a=0.0,
            E0=130e9,
            nu=0.34,
            T_melt=1357,
            M_amu=63.546,
            rho=8960,
            delta_L=0.10,
            # Λ³ fitted parameters
            lambda_base=26.3,
            kappa=1.713,
        )
    
    @classmethod
    def FCC_Al(cls):
        """FCCアルミニウム"""
        return cls(
            name="FCC-Al",
            structure="FCC",
            Z_bulk=12,
            a_300K=4.05e-10,
            alpha=2.3e-5,
            c_over_a=0.0,
            E0=70e9,
            nu=0.33,
            T_melt=933,
            M_amu=26.982,
            rho=2700,
            delta_L=0.11,
            # Λ³ fitted parameters
            lambda_base=27.3,
            kappa=4.180,
        )
    
    @classmethod
    def HCP_Ti(cls):
        """HCPチタン"""
        return cls(
            name="HCP-Ti",
            structure="HCP",
            Z_bulk=12,
            a_300K=2.95e-10,
            alpha=8.6e-6,
            c_over_a=1.587,
            E0=116e9,
            nu=0.32,
            T_melt=1941,
            M_amu=47.867,
            rho=4506,
            delta_L=0.10,
            # Λ³ fitted parameters
            lambda_base=43.1,
            kappa=0.771,
        )
    
    @classmethod
    def HCP_Mg(cls):
        """HCPマグネシウム"""
        return cls(
            name="HCP-Mg",
            structure="HCP",
            Z_bulk=12,
            a_300K=3.21e-10,
            alpha=2.6e-5,  # フィッティングデータに合わせる
            c_over_a=1.624,
            E0=45e9,
            nu=0.29,
            T_melt=923,
            M_amu=24.305,
            rho=1738,
            delta_L=0.117,
            # Λ³ fitted parameters (Mg has high κ due to anharmonicity)
            lambda_base=7.5,
            kappa=37.568,
        )


class PhysicsEngine:
    """
    Λ³物理エンジン
    
    動力学 × 幾何学 = Λ³-Dynamics
    
    Core equations:
      U²_c ∝ Z³  (critical displacement)
      f_G  ∝ Z³  (Born collapse factor)
      
    Components:
      - Born collapse (thermal softening)
      - Debye-Waller (thermal vibration)
      - Hooke (mechanical strain)
      - Lindemann criterion (stability)
    """
    
    # Z³ scaling constants (from 7-metal validation)
    FG_FCC_REF = 0.097   # Reference: FCC at Z=12
    Z_FCC_REF = 12       # Reference coordination
    
    def __init__(self, material: MaterialPhysics):
        self.mat = material
        self.M = material.M_amu * u_kg  # kg
        
        # 室温の弾性定数
        self.G0 = material.E0 / (2.0 * (1.0 + material.nu))  # 剛性率
        self.K0 = material.E0 / (3.0 * (1.0 - 2.0 * material.nu))  # 体積弾性率
        
        # Z³ベースのfG計算
        self.Z_eff = self._compute_Z_eff()
        self.fG_at_melt = self._compute_fG_Z3()
        
        print("="*60)
        print("Λ³ Physics Engine Initialized")
        print("="*60)
        print(f"Material: {material.name} ({material.structure})")
        print(f"a(300K) = {material.a_300K*1e10:.3f} Å")
        if material.c_over_a > 0:
            print(f"c/a = {material.c_over_a:.3f}")
        print(f"E₀ = {material.E0/1e9:.1f} GPa")
        print(f"G₀ = {self.G0/1e9:.2f} GPa")
        print(f"K₀ = {self.K0/1e9:.2f} GPa")
        print(f"T_melt = {material.T_melt} K")
        print(f"δ_L = {material.delta_L}")
        print(f"Z_eff = {self.Z_eff:.2f} (Z³ scaling)")
        print(f"f_G(T_melt) = {self.fG_at_melt:.4f} (Z³ predicted)")
    
    def _compute_Z_eff(self) -> float:
        """
        有効配位数を計算
        
        BCC: Z_eff = 8
        FCC: Z_eff = 12
        HCP: Z_eff = 6 + 6×g(c/a), g depends on c/a ratio
        """
        if self.mat.structure == 'BCC':
            return 8.0
        elif self.mat.structure == 'FCC':
            return 12.0
        elif self.mat.structure == 'HCP':
            # c/a補正: ideal = 1.633
            c_a_ideal = 1.633
            c_a_ratio = self.mat.c_over_a / c_a_ideal
            
            # 経験的モデル: c/aがidealから外れるとZ_eff低下
            # Ti (c/a=1.587): Z_eff ≈ 11.3
            # Mg (c/a=1.624): Z_eff ≈ 11.1
            g = 0.85 + 0.08 * c_a_ratio  # 簡易線形モデル
            g = min(max(g, 0.80), 1.00)  # [0.80, 1.00]にクリップ
            
            return 6.0 + 6.0 * g
        else:
            return 12.0  # デフォルト
    
    def _compute_fG_Z3(self) -> float:
        """
        Z³スケーリングでBorn collapse係数を計算
        
        f_G = f_G_ref × (Z_eff / Z_ref)³
        
        Reference: FCC (Z=12), f_G = 0.097
        
        Results:
          BCC (Z=8):  f_G ≈ 0.029
          FCC (Z=12): f_G ≈ 0.097
          HCP (Z≈11): f_G ≈ 0.079
        """
        return self.FG_FCC_REF * (self.Z_eff / self.Z_FCC_REF) ** 3
    
    # ========================================
    # 1. 幾何関数（温度依存）
    # ========================================
    
    def lattice_constant(self, T: float) -> float:
        """
        温度依存の格子定数 a(T)
        
        a(T) = a₀ × (1 + α × (T - T_ref))
        """
        return self.mat.a_300K * (1.0 + self.mat.alpha * (T - 300.0))
    
    def nearest_neighbor_distance(self, T: float) -> float:
        """
        最近接原子間距離 r_nn(T)
        
        BCC: r_nn = a√3/2
        FCC: r_nn = a/√2
        HCP: r_nn = a (basal plane, simplified)
        """
        a = self.lattice_constant(T)
        if self.mat.structure == "BCC":
            return a * math.sqrt(3) / 2
        elif self.mat.structure == "FCC":
            return a / math.sqrt(2)
        elif self.mat.structure == "HCP":
            # Basal plane nearest neighbor = a
            # (c-axis neighbors are slightly different but we use basal for simplicity)
            return a
        return a / math.sqrt(2)  # デフォルト
    
    def number_density(self, T: float) -> float:
        """
        原子数密度 n(T) [atoms/m³]
        
        BCC: 2 atoms per unit cell, V = a³
        FCC: 4 atoms per unit cell, V = a³
        HCP: 2 atoms per unit cell, V = (√3/2) × a² × c
        """
        a = self.lattice_constant(T)
        if self.mat.structure == "BCC":
            return 2.0 / (a**3)
        elif self.mat.structure == "FCC":
            return 4.0 / (a**3)
        elif self.mat.structure == "HCP":
            c = a * self.mat.c_over_a
            V_cell = (math.sqrt(3.0) / 2.0) * a**2 * c
            return 2.0 / V_cell
        return 4.0 / (a**3)  # デフォルト
    
    # ========================================
    # 2. Born Collapse（剛性率の温度崩壊）
    #    Z³スケーリング統合版
    # ========================================
    
    def born_collapse_factor(self, T: float) -> float:
        """
        Born collapse係数 fG(T) - Λ³熱軟化モデル
        
        E(T)/E₀ = exp[-λ_eff × α × ΔT]
        
        where:
          λ_eff = λ_base × (1 + κ × ΔT/1000)
          ΔT = T - T_ref (T_ref = 293K)
        
        Parameters from 7-material fitting:
          Fe:  λ=49.2, κ=0.573  (Residual 1.5%)
          Cu:  λ=26.3, κ=1.713  (Residual 0.3%)
          Al:  λ=27.3, κ=4.180  (Residual 1.1%)
          Ti:  λ=43.1, κ=0.771  (Residual 0.3%)
          Mg:  λ=7.5,  κ=37.57  (Residual 2.1%)
        """
        T_ref = 293.0  # 参照温度 [K]
        
        if T <= T_ref:
            return 1.0
        
        delta_T = T - T_ref
        
        # 非調和補正付き有効λ
        lambda_eff = self.mat.lambda_base * (1.0 + self.mat.kappa * delta_T / 1000.0)
        
        # Λ³熱軟化
        fG = math.exp(-lambda_eff * self.mat.alpha * delta_T)
        
        # Z³スケーリングの下限を保証（融点では少なくともfG_at_melt）
        return max(fG, self.fG_at_melt)
    
    def shear_modulus(self, T: float) -> float:
        """
        温度依存の剛性率 G(T)
        
        G(T) = G₀ × fG(T)
        """
        fG = self.born_collapse_factor(T)
        return self.G0 * fG
    
    def bulk_modulus(self, T: float) -> float:
        """
        温度依存の体積弾性率 K(T)
        
        体積弾性率は剛性率ほど崩壊しない
        K(T) ≈ K₀ × (1 - 0.3 × (T/T_melt)²)
        """
        T_ratio = min(T / self.mat.T_melt, 1.0)
        return self.K0 * (1.0 - 0.3 * T_ratio**2)
    
    def youngs_modulus(self, T: float) -> float:
        """
        温度依存のヤング率 E(T)
        
        E = 9KG / (3K + G)
        """
        G = self.shear_modulus(T)
        K = self.bulk_modulus(T)
        return 9.0 * K * G / (3.0 * K + G)
    
    # ========================================
    # 3. Debye-Waller（熱的原子振動）
    # ========================================
    
    def sound_velocities(self, T: float) -> Tuple[float, float]:
        """
        音速 v_t（横波）、v_l（縦波）
        
        v_t = √(G/ρ)
        v_l = √((K + 4G/3)/ρ)
        """
        G = self.shear_modulus(T)
        K = self.bulk_modulus(T)
        n = self.number_density(T)
        rho = n * self.M  # 実際の密度
        
        v_t = math.sqrt(G / rho)
        v_l = math.sqrt((K + 4.0*G/3.0) / rho)
        
        return v_t, v_l
    
    def debye_wavevector(self, T: float) -> float:
        """
        Debye波数 k_D
        
        k_D = (6π²n)^(1/3)
        """
        n = self.number_density(T)
        return (6.0 * math.pi**2 * n) ** (1.0/3.0)
    
    def inverse_omega_squared(self, T: float) -> float:
        """
        ⟨1/ω²⟩の計算（Debye模型）
        
        ⟨1/ω²⟩ = (1/3k_D²) × (2/v_t² + 1/v_l²)
        """
        v_t, v_l = self.sound_velocities(T)
        k_D = self.debye_wavevector(T)
        
        inv_omega2 = (1.0 / (3.0 * k_D**2)) * (2.0/v_t**2 + 1.0/v_l**2)
        return inv_omega2
    
    def thermal_displacement_squared(self, T: float) -> float:
        """
        熱的原子変位の二乗 ⟨u²⟩_thermal
        
        ⟨u²⟩ = (k_B T / M) × ⟨1/ω²⟩
        
        これがDebye-Waller因子の元！
        """
        if T <= 0:
            return 0.0
        
        inv_omega2 = self.inverse_omega_squared(T)
        u2_thermal = (k_B * T / self.M) * inv_omega2
        
        return u2_thermal
    
    def thermal_lindemann_ratio(self, T: float) -> float:
        """
        熱的Lindemann比 δ_thermal
        
        δ_thermal = √⟨u²⟩ / r_nn
        """
        u2 = self.thermal_displacement_squared(T)
        r_nn = self.nearest_neighbor_distance(T)
        
        return math.sqrt(u2) / r_nn
    
    # ========================================
    # 4. Hooke（機械的変形）
    # ========================================
    
    def mechanical_displacement_squared(self, 
                                          strain_tensor: np.ndarray,
                                          T: float = 300.0) -> np.ndarray:
        """
        機械的原子変位の二乗 U²_mech
        
        ひずみテンソル E から原子スケールの変位を計算
        
        修正版：主ひずみ（最大固有値）を使用
        理由：原子は主に最大ひずみ方向に変位する
        
        U²_mech = ε_max² × r_nn²
        
        Args:
            strain_tensor: Green-Lagrangeひずみ [N, 3, 3] or [3, 3]
            T: 温度 [K]
        
        Returns:
            U²_mech [N] or scalar
        """
        r_nn = self.nearest_neighbor_distance(T)
        
        if strain_tensor.ndim == 2:
            # 単一テンソル
            eigvals = np.linalg.eigvalsh(strain_tensor)
            eps_max = np.max(np.abs(eigvals))
            U2_mech = (eps_max * r_nn)**2
        else:
            # N個のテンソル
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
        """
        機械的Lindemann比 δ_mech
        
        δ_mech = √U²_mech / r_nn = ||E||
        """
        U2_mech = self.mechanical_displacement_squared(strain_tensor, T)
        r_nn = self.nearest_neighbor_distance(T)
        
        return np.sqrt(U2_mech) / r_nn
    
    # ========================================
    # 5. 統合Λ³計算
    # ========================================
    
    def total_U2(self, 
                  strain_tensor: np.ndarray,
                  T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        総U²を計算
        
        U²_total = U²_thermal + U²_mech
        
        Returns:
            U2_total, U2_thermal, U2_mech
        """
        U2_thermal = self.thermal_displacement_squared(T)
        U2_mech = self.mechanical_displacement_squared(strain_tensor, T)
        
        # U²_thermalはスカラー、U²_mechは配列
        if isinstance(U2_mech, np.ndarray):
            U2_thermal_arr = np.full_like(U2_mech, U2_thermal)
        else:
            U2_thermal_arr = U2_thermal
        
        U2_total = U2_thermal_arr + U2_mech
        
        return U2_total, U2_thermal_arr, U2_mech
    
    def critical_U2(self, Z_eff: np.ndarray = None) -> np.ndarray:
        """
        臨界U²_c を計算
        
        U²_c = (δ_L × r_nn)² × (Z_eff / Z_bulk)³
        
        Args:
            Z_eff: 有効配位数 [N] or scalar
        """
        r_nn = self.nearest_neighbor_distance(300.0)  # 参照温度で
        
        U2_c_bulk = (self.mat.delta_L * r_nn)**2
        
        if Z_eff is None:
            return U2_c_bulk
        
        Z_ratio = np.asarray(Z_eff) / self.mat.Z_bulk
        U2_c = U2_c_bulk * (Z_ratio ** 3)
        
        return U2_c
    
    def compute_lambda(self,
                        strain_tensor: np.ndarray,
                        T: float,
                        Z_eff: np.ndarray = None) -> np.ndarray:
        """
        λ = U²_total / U²_c を計算
        """
        U2_total, _, _ = self.total_U2(strain_tensor, T)
        U2_c = self.critical_U2(Z_eff)
        
        return U2_total / np.maximum(U2_c, 1e-30)
    
    # ========================================
    # 6. 診断・レポート
    # ========================================
    
    def report_thermal_state(self, T: float):
        """温度状態をレポート"""
        print(f"\n--- Thermal State at T = {T} K ---")
        
        # 幾何
        a = self.lattice_constant(T)
        r_nn = self.nearest_neighbor_distance(T)
        print(f"a(T) = {a*1e10:.4f} Å")
        print(f"r_nn(T) = {r_nn*1e10:.4f} Å")
        
        # Born collapse
        fG = self.born_collapse_factor(T)
        G = self.shear_modulus(T)
        E = self.youngs_modulus(T)
        print(f"fG(T) = {fG:.3f} (Born collapse)")
        print(f"G(T) = {G/1e9:.2f} GPa")
        print(f"E(T) = {E/1e9:.1f} GPa")
        
        # Debye-Waller
        v_t, v_l = self.sound_velocities(T)
        u2 = self.thermal_displacement_squared(T)
        delta = self.thermal_lindemann_ratio(T)
        print(f"v_t = {v_t:.0f} m/s, v_l = {v_l:.0f} m/s")
        print(f"⟨u²⟩_thermal = {u2:.4e} m²")
        print(f"√⟨u²⟩ = {math.sqrt(u2)*1e10:.4f} Å")
        print(f"δ_thermal = {delta:.4f}")
        
        # 臨界判定
        if delta >= self.mat.delta_L:
            print(f"⚠️  δ ≥ δ_L = {self.mat.delta_L} → UNSTABLE!")
        else:
            print(f"✓ δ < δ_L = {self.mat.delta_L} → stable")
    
    def validate_lindemann_at_melting(self):
        """融点でLindemann判定を検証"""
        print(f"\n{'='*60}")
        print(f"Lindemann Validation at T_melt = {self.mat.T_melt} K")
        print(f"{'='*60}")
        
        delta = self.thermal_lindemann_ratio(self.mat.T_melt)
        
        print(f"δ_thermal(T_melt) = {delta:.4f}")
        print(f"δ_L (target) = {self.mat.delta_L}")
        
        if 0.08 <= delta <= 0.20:
            print(f"✅ Lindemann criterion validated!")
        else:
            print(f"⚠️ δ = {delta:.3f} outside typical range [0.08, 0.20]")
        
        return delta


# ========================================
# テスト
# ========================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Λ³ PHYSICS ENGINE TEST (Z³ Scaling Version)")
    print("="*70)
    
    # 全構造をテスト
    materials = [
        ("BCC-Fe", MaterialPhysics.BCC_Fe()),
        ("FCC-Cu", MaterialPhysics.FCC_Cu()),
        ("FCC-Al", MaterialPhysics.FCC_Al()),
        ("HCP-Ti", MaterialPhysics.HCP_Ti()),
        ("HCP-Mg", MaterialPhysics.HCP_Mg()),
    ]
    
    results = []
    
    for name, mat in materials:
        print(f"\n{'='*60}")
        engine = PhysicsEngine(mat)
        delta_pred = engine.thermal_lindemann_ratio(mat.T_melt)
        delta_exp = mat.delta_L
        error = abs(delta_pred - delta_exp) / delta_exp * 100
        
        results.append({
            'name': name,
            'struct': mat.structure,
            'Z_eff': engine.Z_eff,
            'fG': engine.fG_at_melt,
            'delta_pred': delta_pred,
            'delta_exp': delta_exp,
            'error': error,
        })
        
        print(f"\nValidation at T_melt = {mat.T_melt} K:")
        print(f"  δ_pred = {delta_pred:.4f}")
        print(f"  δ_exp  = {delta_exp:.4f}")
        print(f"  Error  = {error:.1f}%")
    
    # サマリーテーブル
    print("\n" + "="*70)
    print("Z³ SCALING SUMMARY")
    print("="*70)
    print(f"{'Material':<10} {'Struct':<6} {'Z_eff':<8} {'fG':<8} {'δ_pred':<8} {'δ_exp':<8} {'Error%':<8}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<10} {r['struct']:<6} {r['Z_eff']:<8.2f} {r['fG']:<8.4f} "
              f"{r['delta_pred']:<8.4f} {r['delta_exp']:<8.4f} {r['error']:<8.1f}")
    
    mean_error = np.mean([r['error'] for r in results])
    print("-"*70)
    print(f"Mean Absolute Error: {mean_error:.1f}%")
    
    # Z³スケーリングの検証
    print("\n" + "="*70)
    print("Z³ SCALING VERIFICATION")
    print("="*70)
    print(f"Reference: FCC (Z=12), fG = {PhysicsEngine.FG_FCC_REF}")
    print(f"\nfG = 0.097 × (Z_eff/12)³")
    print(f"\n{'Structure':<10} {'Z_eff':<8} {'fG_pred':<10} {'(Z/12)³':<10}")
    for struct, Z in [('BCC', 8), ('FCC', 12), ('HCP', 11.2)]:
        fG_pred = 0.097 * (Z/12)**3
        Z_ratio = (Z/12)**3
        print(f"{struct:<10} {Z:<8.1f} {fG_pred:<10.4f} {Z_ratio:<10.4f}")
