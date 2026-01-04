"""
Λ-Dynamics Material Database
=============================
材料の物性パラメータを定義

SECD: 電気亜鉛めっき鋼板（ベースはSPCD系）
"""

import numpy as np

# 物理定数
PHYSICAL_CONSTANTS = {
    'k_B': 1.380649e-23,    # Boltzmann定数 [J/K]
    'hbar': 1.054572e-34,   # Dirac定数 [J·s]
    'eV_to_J': 1.602176e-19, # eV → J
    'amu': 1.660539e-27,    # 原子質量単位 [kg]
}

# 純金属データベース（7金属）
PURE_METALS = {
    'Fe': {
        'name': 'Iron',
        'structure': 'BCC',
        'Z_bulk': 8,
        'a': 2.87e-10,           # 格子定数 [m]
        'Tm': 1811,              # 融点 [K]
        'E0': 210e9,             # Young率 [Pa]
        'rho': 7870,             # 密度 [kg/m³]
        'mass': 55.845 * 1.660539e-27,  # 原子質量 [kg]
        'alpha': 15e-6,          # 熱膨張係数 [K⁻¹]
        'lambda_base': 49.2,     # 熱軟化係数
        'kappa': 0.573,          # 非調和性
        'fG': 0.027,             # Born崩壊係数（BCC）
        'delta_L': 0.180,        # Lindemann比
        'bond_energy_eV': 4.28,  # 結合エネルギー [eV]
    },
    'Cu': {
        'name': 'Copper',
        'structure': 'FCC',
        'Z_bulk': 12,
        'a': 3.615e-10,
        'Tm': 1357,
        'E0': 130e9,
        'rho': 8960,
        'mass': 63.546 * 1.660539e-27,
        'alpha': 17e-6,
        'lambda_base': 26.3,
        'kappa': 1.713,
        'fG': 0.101,             # FCC
        'delta_L': 0.100,
        'bond_energy_eV': 3.49,
    },
    'Al': {
        'name': 'Aluminum',
        'structure': 'FCC',
        'Z_bulk': 12,
        'a': 4.05e-10,
        'Tm': 933,
        'E0': 70e9,
        'rho': 2700,
        'mass': 26.982 * 1.660539e-27,
        'alpha': 24e-6,
        'lambda_base': 27.3,
        'kappa': 4.18,
        'fG': 0.101,
        'delta_L': 0.100,
        'bond_energy_eV': 3.39,
    },
}

# 工業用合金データベース
INDUSTRIAL_ALLOYS = {
    'SECD': {
        # 電気亜鉛めっき鋼板（ベースはSPCD系）
        'name': 'Electrogalvanized Steel (Drawing Quality)',
        'base': 'Fe',
        'structure': 'BCC',
        'Z_bulk': 8,
        'a': 2.87e-10,
        'Tm': 1811,
        'E0': 210e9,
        'rho': 7870,
        'mass': 55.845 * 1.660539e-27,
        'alpha': 15e-6,
        'lambda_base': 49.2,
        'kappa': 0.573,
        'fG': 0.027,
        'delta_L': 0.180,
        'bond_energy_eV': 4.28,
        # プレス成形用パラメータ（JIS G 3313準拠）
        'n_value': 0.20,         # 加工硬化指数（0.18-0.22の中間）
        'r_value': 1.45,         # ランクフォード値（1.3-1.6の中間）
        'yield_stress': 160e6,   # 降伏応力 [Pa]（140-180の中間）
        'tensile_strength': 320e6,  # 引張強さ [Pa]（270-370の中間）
        'elongation': 0.36,      # 破断伸び（34%以上）
        # FLC実データ（SECD t=1.96mm）
        # 形式: (β = ε₂/ε₁, ε₁_limit)
        'flc_points': [
            (-0.370, 0.540),  # 単軸引張 (Uniaxial)
            (-0.306, 0.490),  # 深絞り (Deep Draw)
            (-0.169, 0.415),  # 中間1 (Draw-Plane)
            ( 0.000, 0.346),  # 平面ひずみ (Plane Strain) ← FLC₀
            ( 0.133, 0.375),  # 中間2 (Plane-Stretch)
            ( 0.247, 0.405),  # 張出し (Stretch)
            ( 0.430, 0.465),  # 等二軸 (Equi-Biaxial)
        ],
        'flc_t_ref': 1.96,       # FLC参照板厚 [mm]
        # Keeler式パラメータ（板厚補正用）
        'flc_a': 23.3,           # FLC₀ = a + b × t
        'flc_b': 14.1,           # [%/mm]
    },
    'SPCC': {
        # 冷間圧延鋼板（一般用）
        'name': 'Cold Rolled Steel (Commercial Quality)',
        'base': 'Fe',
        'structure': 'BCC',
        'Z_bulk': 8,
        'a': 2.87e-10,
        'Tm': 1811,
        'E0': 210e9,
        'rho': 7870,
        'mass': 55.845 * 1.660539e-27,
        'alpha': 15e-6,
        'lambda_base': 49.2,
        'kappa': 0.573,
        'fG': 0.027,
        'delta_L': 0.180,
        'bond_energy_eV': 4.28,
        'n_value': 0.18,
        'r_value': 1.2,
        'yield_stress': 220e6,
        'tensile_strength': 370e6,
    },
}


class Material:
    """材料クラス"""
    
    def __init__(self, name: str):
        """
        Args:
            name: 材料名（'Fe', 'SECD'など）
        """
        if name in PURE_METALS:
            self.data = PURE_METALS[name].copy()
        elif name in INDUSTRIAL_ALLOYS:
            self.data = INDUSTRIAL_ALLOYS[name].copy()
        else:
            raise ValueError(f"Unknown material: {name}")
        
        self.name = name
        self.k_B = PHYSICAL_CONSTANTS['k_B']
        
        # 導出パラメータを計算
        self._compute_derived_params()
    
    def _compute_derived_params(self):
        """導出パラメータを計算"""
        # Debye周波数の近似（Einstein model）
        # ω ≈ sqrt(E / (ρ × a²))
        E = self.data['E0']
        rho = self.data['rho']
        a = self.data['a']
        self.omega = np.sqrt(E / (rho * a**2))
        
        # 結合エネルギー [J]
        self.bond_energy_J = self.data['bond_energy_eV'] * PHYSICAL_CONSTANTS['eV_to_J']
        
        # 基準U²_c（バルク、室温）
        delta_L = self.data['delta_L']
        self.U2_c_bulk = (delta_L * a)**2
        
    def thermal_softening(self, T: np.ndarray) -> np.ndarray:
        """
        熱軟化係数を計算
        
        E(T) / E₀ = exp[-λ_eff × α × ΔT]
        
        Args:
            T: 温度 [K]
        
        Returns:
            E(T) / E₀
        """
        T_ref = 293.0  # 室温
        alpha = self.data['alpha']
        lambda_base = self.data['lambda_base']
        kappa = self.data['kappa']
        
        delta_T = T - T_ref
        lambda_eff = lambda_base * (1 + kappa * delta_T / 1000)
        
        return np.exp(-lambda_eff * alpha * delta_T)
    
    def U2_critical(self, Z: np.ndarray, T: np.ndarray = None) -> np.ndarray:
        """
        臨界U²を計算（Z³スケーリング + 熱軟化）
        
        U²_c = U²_c_bulk × fG / (Z/Z_bulk)³ × f_thermal(T)
        
        Args:
            Z: 配位数
            T: 温度 [K]（Noneなら室温）
        
        Returns:
            U²_c [m²]
        """
        Z = np.asarray(Z, dtype=float)
        Z_bulk = self.data['Z_bulk']
        fG = self.data['fG']
        
        # Z³スケーリング
        Z_ratio = np.maximum(Z / Z_bulk, 0.01)  # ゼロ除算防止
        Z_factor = Z_ratio**3
        
        U2_c = self.U2_c_bulk * fG / Z_factor
        
        # 熱軟化（温度が指定された場合）
        if T is not None:
            thermal_factor = self.thermal_softening(np.asarray(T))
            U2_c = U2_c * thermal_factor
        
        return U2_c
    
    def U2_thermal(self, T: np.ndarray) -> np.ndarray:
        """
        熱振動によるU²を計算
        
        U²_thermal = k_B × T / (m × ω²)
        
        Args:
            T: 温度 [K]
        
        Returns:
            U²_thermal [m²]
        """
        T = np.asarray(T, dtype=float)
        m = self.data['mass']
        
        return self.k_B * T / (m * self.omega**2)
    
    def __repr__(self):
        return f"Material('{self.name}', structure={self.data['structure']}, E0={self.data['E0']/1e9:.0f}GPa)"
    
    def compute_FLC0(self, thickness_mm: float) -> float:
        """
        FLC₀（平面ひずみ限界）を計算
        
        実データがあればそこから、なければKeeler-Brazier式
        板厚が変化した場合は補正
        
        Args:
            thickness_mm: 板厚 [mm]
        
        Returns:
            FLC₀ [fraction, not %]
        """
        if 'flc_points' in self.data:
            # 実データからFLC₀を取得（β=0のポイント）
            flc_points = self.data['flc_points']
            for beta, eps1_limit in flc_points:
                if abs(beta) < 0.01:  # β ≈ 0
                    flc0_ref = eps1_limit
                    break
            else:
                # β=0がなければ補間
                flc0_ref = self._interpolate_flc(0.0)
            
            # 板厚補正（参照板厚からの変化）
            t_ref = self.data.get('flc_t_ref', thickness_mm)
            if t_ref > 0 and thickness_mm != t_ref:
                # FLC₀ ∝ √t の近似補正（薄くなると限界が下がる）
                thickness_factor = np.sqrt(thickness_mm / t_ref)
                flc0 = flc0_ref * thickness_factor
            else:
                flc0 = flc0_ref
            
            return flc0
        else:
            # Keeler-Brazier式
            a = self.data.get('flc_a', 23.3)
            b = self.data.get('flc_b', 14.1)
            flc0_percent = a + b * thickness_mm
            return flc0_percent / 100.0
    
    def _interpolate_flc(self, beta: float) -> float:
        """FLCデータを補間"""
        if 'flc_points' not in self.data:
            return 0.35  # デフォルト
        
        flc_points = sorted(self.data['flc_points'], key=lambda x: x[0])
        betas = [p[0] for p in flc_points]
        eps1s = [p[1] for p in flc_points]
        
        # 線形補間
        return np.interp(beta, betas, eps1s)
    
    def compute_FLC(self, thickness_mm: float, minor_strain: np.ndarray) -> np.ndarray:
        """
        FLC（成形限界曲線）を計算 - 実データ補間版
        
        Args:
            thickness_mm: 板厚 [mm]（スカラーまたは配列）
            minor_strain: マイナーひずみ ε₂
        
        Returns:
            major_strain_limit: メジャーひずみ限界 ε₁
        """
        minor_strain = np.asarray(minor_strain)
        thickness_mm = np.asarray(thickness_mm)
        
        if 'flc_points' in self.data:
            # 実データから補間
            flc_points = sorted(self.data['flc_points'], key=lambda x: x[0])
            betas = np.array([p[0] for p in flc_points])
            eps1s = np.array([p[1] for p in flc_points])
            
            # 各点のβを計算（ε₂ / ε₁_limit の近似）
            # 簡略化: minor_strain 自体をβとして使用
            # （実際は ε₁ が未知なので反復計算が必要だが、ここでは近似）
            
            # β ≈ ε₂ / ε₁_estimated
            # 初期推定: FLC₀付近を仮定
            flc0 = self._interpolate_flc(0.0)
            beta_estimated = minor_strain / np.maximum(np.abs(minor_strain) + flc0, 0.01)
            beta_estimated = np.clip(beta_estimated, betas.min(), betas.max())
            
            # FLC値を補間
            major_limit_ref = np.interp(beta_estimated, betas, eps1s)
            
            # 板厚補正
            t_ref = self.data.get('flc_t_ref', 1.96)
            if np.isscalar(thickness_mm):
                thickness_factor = np.sqrt(thickness_mm / t_ref)
            else:
                thickness_factor = np.sqrt(np.maximum(thickness_mm, 0.1) / t_ref)
            
            major_limit = major_limit_ref * thickness_factor
            
            return major_limit
        else:
            # 従来の線形近似
            flc0 = self.compute_FLC0(thickness_mm if np.isscalar(thickness_mm) else thickness_mm.mean())
            left_slope = self.data.get('flc_left_slope', -1.0)
            right_slope = self.data.get('flc_right_slope', 0.55)
            
            major_limit = np.where(
                minor_strain < 0,
                flc0 - left_slope * minor_strain,
                flc0 + right_slope * minor_strain
            )
            
            return major_limit
    
    def strain_to_lambda(self, strain: np.ndarray, thickness_mm: float = 1.96,
                         strain_ratio: float = 0.0) -> np.ndarray:
        """
        工学的ひずみをλ（U²/U²_c）に変換
        
        FLCを基準にスケーリング：
        - ε = FLC₀ のとき λ = 1（破断）
        
        Args:
            strain: 等価ひずみ or メジャーひずみ
            thickness_mm: 板厚 [mm]
            strain_ratio: ひずみ比 β = ε₂/ε₁（0=平面ひずみ）
        
        Returns:
            λ = U²/U²_c
        """
        strain = np.asarray(strain)
        
        # FLC₀を取得
        flc0 = self.compute_FLC0(thickness_mm)
        
        # ひずみ比を考慮したFLC限界
        minor_strain = strain * strain_ratio
        flc_limit = self.compute_FLC(thickness_mm, minor_strain)
        
        # λ = (ε / ε_limit)²
        # 2乗するのは U² ∝ ε² の関係から
        lambda_val = (np.abs(strain) / flc_limit) ** 2
        
        return lambda_val


if __name__ == "__main__":
    # テスト
    secd = Material('SECD')
    print(secd)
    print(f"  Debye freq: {secd.omega:.2e} rad/s")
    print(f"  U²_c (bulk): {secd.U2_c_bulk:.2e} m²")
    print(f"  U²_thermal (300K): {secd.U2_thermal(300):.2e} m²")
    
    # Z³スケーリングのテスト
    Z_values = np.array([8, 6, 4, 2])  # bulk, surface, edge, corner
    U2_c = secd.U2_critical(Z_values)
    print(f"\nZ³ scaling:")
    for z, u2c in zip(Z_values, U2_c):
        print(f"  Z={z}: U²_c = {u2c:.2e} m²  (ratio to bulk: {u2c/U2_c[0]:.2f})")
