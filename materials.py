"""
Λ³-Dynamics Material Database
=============================

材料の物性パラメータのみを定義（計算ロジックはphysics_engine.pyへ）

使用法:
  from materials import get_material
  mat = get_material('Fe')  # dict
  mat = get_material('SECD')  # dict
"""

# ============================================================
# 物理定数
# ============================================================
PHYSICAL_CONSTANTS = {
    'k_B': 1.380649e-23,     # Boltzmann定数 [J/K]
    'hbar': 1.054572e-34,    # Dirac定数 [J·s]
    'eV_to_J': 1.602176e-19, # eV → J
    'amu': 1.660539e-27,     # 原子質量単位 [kg]
}

# ============================================================
# 純金属データベース（7金属、Λ³熱軟化パラメータ込み）
# ============================================================
PURE_METALS = {
    'Fe': {
        'name': 'Iron',
        'structure': 'BCC',
        'Z_bulk': 8,
        'a': 2.87e-10,           # 格子定数 [m]
        'c_over_a': 0.0,         # HCP用（BCC/FCCは0）
        'T_melt': 1811,          # 融点 [K]
        'E0': 210e9,             # ヤング率 [Pa]
        'nu': 0.29,              # ポアソン比
        'rho': 7870,             # 密度 [kg/m³]
        'M_amu': 55.845,         # 原子量 [amu]
        'alpha': 1.5e-5,         # 熱膨張係数 [1/K]
        # Λ³熱軟化パラメータ（7材料フィッティング済み）
        'lambda_base': 49.2,     # 基準減衰係数
        'kappa': 0.573,          # 非調和パラメータ
        # Lindemann
        'delta_L': 0.180,        # Lindemann比
        # 結合エネルギー
        'bond_energy_eV': 4.28,
    },
    'Cu': {
        'name': 'Copper',
        'structure': 'FCC',
        'Z_bulk': 12,
        'a': 3.615e-10,
        'c_over_a': 0.0,
        'T_melt': 1357,
        'E0': 130e9,
        'nu': 0.34,
        'rho': 8960,
        'M_amu': 63.546,
        'alpha': 1.7e-5,
        'lambda_base': 26.3,
        'kappa': 1.713,
        'delta_L': 0.100,
        'bond_energy_eV': 3.49,
    },
    'Al': {
        'name': 'Aluminum',
        'structure': 'FCC',
        'Z_bulk': 12,
        'a': 4.05e-10,
        'c_over_a': 0.0,
        'T_melt': 933,
        'E0': 70e9,
        'nu': 0.33,
        'rho': 2700,
        'M_amu': 26.982,
        'alpha': 2.3e-5,
        'lambda_base': 27.3,
        'kappa': 4.180,
        'delta_L': 0.110,
        'bond_energy_eV': 3.39,
    },
    'Ni': {
        'name': 'Nickel',
        'structure': 'FCC',
        'Z_bulk': 12,
        'a': 3.52e-10,
        'c_over_a': 0.0,
        'T_melt': 1728,
        'E0': 200e9,
        'nu': 0.31,
        'rho': 8908,
        'M_amu': 58.693,
        'alpha': 1.3e-5,
        'lambda_base': 22.6,
        'kappa': 0.279,
        'delta_L': 0.090,
        'bond_energy_eV': 4.44,
    },
    'Ti': {
        'name': 'Titanium',
        'structure': 'HCP',
        'Z_bulk': 12,
        'a': 2.95e-10,
        'c_over_a': 1.587,       # HCP
        'T_melt': 1941,
        'E0': 116e9,
        'nu': 0.32,
        'rho': 4506,
        'M_amu': 47.867,
        'alpha': 8.6e-6,
        'lambda_base': 43.1,
        'kappa': 0.771,
        'delta_L': 0.100,
        'bond_energy_eV': 4.85,
    },
    'W': {
        'name': 'Tungsten',
        'structure': 'BCC',
        'Z_bulk': 8,
        'a': 3.16e-10,
        'c_over_a': 0.0,
        'T_melt': 3695,
        'E0': 411e9,
        'nu': 0.28,
        'rho': 19250,
        'M_amu': 183.84,
        'alpha': 4.5e-6,
        'lambda_base': 10.9,
        'kappa': 2.759,
        'delta_L': 0.070,
        'bond_energy_eV': 8.90,
    },
    'Mg': {
        'name': 'Magnesium',
        'structure': 'HCP',
        'Z_bulk': 12,
        'a': 3.21e-10,
        'c_over_a': 1.624,       # HCP
        'T_melt': 923,
        'E0': 45e9,
        'nu': 0.29,
        'rho': 1738,
        'M_amu': 24.305,
        'alpha': 2.6e-5,
        'lambda_base': 7.5,
        'kappa': 37.568,         # 高い非調和性
        'delta_L': 0.117,
        'bond_energy_eV': 1.51,
    },
}

# ============================================================
# 工業用合金データベース
# ============================================================
INDUSTRIAL_ALLOYS = {
    'SECD': {
        # 電気亜鉛めっき鋼板（ベースはSPCD系）
        'name': 'Electrogalvanized Steel (Drawing Quality)',
        'base': 'Fe',
        'structure': 'BCC',
        'Z_bulk': 8,
        'a': 2.87e-10,
        'c_over_a': 0.0,
        'T_melt': 1811,
        'E0': 210e9,
        'nu': 0.29,
        'rho': 7870,
        'M_amu': 55.845,
        'alpha': 1.5e-5,
        'lambda_base': 49.2,
        'kappa': 0.573,
        'delta_L': 0.180,
        'bond_energy_eV': 4.28,
        # プレス成形用パラメータ（JIS G 3313準拠）
        'n_value': 0.20,             # 加工硬化指数
        'r_value': 1.45,             # ランクフォード値
        'yield_stress': 160e6,       # 降伏応力 [Pa]
        'tensile_strength': 320e6,   # 引張強さ [Pa]
        'elongation': 0.36,          # 破断伸び
        # FLC実データ（SECD t=1.96mm）
        'flc_points': [
            (-0.370, 0.540),  # 単軸引張 (Uniaxial)
            (-0.306, 0.490),  # 深絞り (Deep Draw)
            (-0.169, 0.415),  # 中間1 (Draw-Plane)
            ( 0.000, 0.346),  # 平面ひずみ (Plane Strain) ← FLC₀
            ( 0.133, 0.375),  # 中間2 (Plane-Stretch)
            ( 0.247, 0.405),  # 張出し (Stretch)
            ( 0.430, 0.465),  # 等二軸 (Equi-Biaxial)
        ],
        'flc_t_ref': 1.96,           # FLC参照板厚 [mm]
    },
    'SPCC': {
        # 冷間圧延鋼板（一般用）
        'name': 'Cold Rolled Steel (Commercial Quality)',
        'base': 'Fe',
        'structure': 'BCC',
        'Z_bulk': 8,
        'a': 2.87e-10,
        'c_over_a': 0.0,
        'T_melt': 1811,
        'E0': 210e9,
        'nu': 0.29,
        'rho': 7870,
        'M_amu': 55.845,
        'alpha': 1.5e-5,
        'lambda_base': 49.2,
        'kappa': 0.573,
        'delta_L': 0.180,
        'bond_energy_eV': 4.28,
        'n_value': 0.18,
        'r_value': 1.2,
        'yield_stress': 220e6,
        'tensile_strength': 370e6,
    },
}


# ============================================================
# ユーティリティ関数
# ============================================================
def get_material(name: str) -> dict:
    """
    材料データを取得
    
    Args:
        name: 材料名（'Fe', 'Cu', 'SECD' など）
    
    Returns:
        材料データのdict
    
    Raises:
        ValueError: 未知の材料名
    """
    if name in PURE_METALS:
        return PURE_METALS[name].copy()
    elif name in INDUSTRIAL_ALLOYS:
        return INDUSTRIAL_ALLOYS[name].copy()
    else:
        available = list(PURE_METALS.keys()) + list(INDUSTRIAL_ALLOYS.keys())
        raise ValueError(f"Unknown material: {name}. Available: {available}")


def list_materials() -> dict:
    """利用可能な材料一覧を返す"""
    return {
        'pure_metals': list(PURE_METALS.keys()),
        'industrial_alloys': list(INDUSTRIAL_ALLOYS.keys()),
    }


# ============================================================
# テスト
# ============================================================
if __name__ == "__main__":
    print("Available materials:")
    print(list_materials())
    
    print("\nPure metals:")
    for name in PURE_METALS:
        mat = get_material(name)
        print(f"  {name}: {mat['structure']}, T_melt={mat['T_melt']}K, δ_L={mat['delta_L']}")
    
    print("\nIndustrial alloys:")
    for name in INDUSTRIAL_ALLOYS:
        mat = get_material(name)
        print(f"  {name}: {mat['name']}")
