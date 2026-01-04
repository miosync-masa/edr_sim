"""
U² Physics Engine - Λ³理論の本質実装
======================================

歪み(ε)を使わない！U²(原子変位²)だけで全て判定

核心:
  λ = U² / U²_c
  
  U² = (r_ij - r₀)²     : 今の原子間距離 - 平衡距離
  U²_c = (δ_L × a)² × (Z/Z_bulk)³  : 臨界U²（Lindemann）

運命の三叉路:
  λ > 1 かつ 引張 → CRACK（ひび）
  λ > 1 かつ 圧縮+熱 → WHITE_LAYER（白層）
  λ > 1 かつ 剪断 → PLASTIC（曲がる）
"""

import numpy as np
from scipy.spatial import cKDTree
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List


class AtomFate(Enum):
    """原子の運命"""
    STABLE = "STABLE"           # λ < 1: 安定
    PLASTIC = "PLASTIC"         # λ > 1 + 剪断: 曲がる（再配列）
    WHITE_LAYER = "WHITE_LAYER" # λ > 1 + 圧縮+熱: 白層
    CRACK = "CRACK"             # λ > 1 + 引張: ひび割れ


@dataclass
class MaterialConstants:
    """材料定数（原子スケール）"""
    name: str
    a: float           # 格子定数 [m]
    r0: float          # 平衡原子間距離 [m] (≈ a/√2 for FCC, a√3/2 for BCC)
    delta_L: float     # Lindemann定数 (≈ 0.1-0.18)
    Z_bulk: int        # バルク配位数 (FCC=12, BCC=8)
    T_melt: float      # 融点 [K]
    
    @classmethod
    def SECD(cls):
        """SECD (低炭素鋼) - BCC"""
        a = 2.87e-10  # m
        return cls(
            name="SECD",
            a=a,
            r0=a * np.sqrt(3) / 2,  # BCC最近接距離
            delta_L=0.18,
            Z_bulk=8,
            T_melt=1811,
        )
    
    @classmethod
    def Aluminum(cls):
        """アルミニウム - FCC"""
        a = 4.05e-10  # m
        return cls(
            name="Al",
            a=a,
            r0=a / np.sqrt(2),  # FCC最近接距離
            delta_L=0.10,
            Z_bulk=12,
            T_melt=933,
        )


class U2PhysicsEngine:
    """
    U²ベースの物理エンジン
    
    歪み(ε)を一切使わない！
    原子変位²と三軸度だけで判定
    """
    
    def __init__(self, material: MaterialConstants):
        """
        Args:
            material: 材料定数
        """
        self.mat = material
        
        # 臨界U²（Lindemann基準）
        self.U2_c_bulk = (self.mat.delta_L * self.mat.a) ** 2
        
        print(f"=== U² Physics Engine ===")
        print(f"Material: {self.mat.name}")
        print(f"a = {self.mat.a*1e10:.3f} Å")
        print(f"r₀ = {self.mat.r0*1e10:.3f} Å")
        print(f"δ_L = {self.mat.delta_L}")
        print(f"U²_c (bulk) = {self.U2_c_bulk:.3e} m²")
        print(f"√U²_c = {np.sqrt(self.U2_c_bulk)*1e10:.3f} Å")
    
    def compute_U2_critical(self, Z: np.ndarray) -> np.ndarray:
        """
        臨界U²を計算（Z³スケーリング）
        
        U²_c = (δ_L × a)² × (Z/Z_bulk)³
        
        表面(Z低) → U²_c低 → 壊れやすい
        """
        Z_ratio = np.clip(Z / self.mat.Z_bulk, 0.1, 2.0)
        return self.U2_c_bulk * (Z_ratio ** 3)
    
    def compute_U2_from_displacement(self, 
                                      vertices_current: np.ndarray,
                                      vertices_reference: np.ndarray) -> np.ndarray:
        """
        変位からU²を計算
        
        U² = |r_current - r_reference|²
        
        Args:
            vertices_current: 現在の頂点座標 [N, 3] (mm)
            vertices_reference: 参照座標 [N, 3] (mm)
        
        Returns:
            U²: 変位の二乗 [N] (m²)
        """
        # mm → m
        disp = (vertices_current - vertices_reference) * 1e-3
        U2 = np.sum(disp**2, axis=1)
        return U2
    
    def compute_U2_from_neighbor_stretch(self,
                                          vertices: np.ndarray,
                                          neighbors: List[np.ndarray],
                                          r0_mm: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        近傍原子との距離変化からU²を計算
        
        dr = r_ij - r₀（材料定数！）
        U² = <dr²> (平均)
        
        さらに体積ひずみ（三軸度の指標）も計算
        
        Args:
            vertices: 頂点座標 [N, 3] (mm)
            neighbors: 各頂点の近傍インデックスリスト
            r0_mm: 平衡距離 (mm)。Noneなら材料定数から計算
        
        Returns:
            U2: 変位の二乗 [N] (m²)
            volumetric_strain: 体積ひずみ [N] (正=膨張=引張, 負=収縮=圧縮)
        """
        n_vertices = len(vertices)
        U2 = np.zeros(n_vertices)
        volumetric_strain = np.zeros(n_vertices)
        
        # ★重要: r₀は材料定数（理想格子の距離）であって、
        # 今の形状から決めてはダメ！
        # でもメッシュの近傍距離はマクロスケール（mm）なので
        # 「平均近傍距離」を理想状態として、それからの偏差を見る
        
        # 近傍距離の分布を取得
        all_distances = []
        for i in range(min(2000, n_vertices)):
            if len(neighbors[i]) > 0:
                dists = np.linalg.norm(vertices[neighbors[i]] - vertices[i], axis=1)
                all_distances.extend(dists)
        
        if len(all_distances) == 0:
            return U2, volumetric_strain
        
        # 「理想状態」= 最も頻度の高い距離（モード）
        # これより伸びてれば引張、縮んでれば圧縮
        hist, bin_edges = np.histogram(all_distances, bins=50)
        mode_idx = np.argmax(hist)
        r0_ideal = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        
        if r0_mm is not None:
            r0_ideal = r0_mm
        
        r0 = r0_ideal * 1e-3  # mm → m
        
        print(f"  r₀ (ideal neighbor distance): {r0_ideal:.3f} mm")
        
        for i in range(n_vertices):
            if len(neighbors[i]) == 0:
                continue
            
            # 近傍との距離
            neighbor_positions = vertices[neighbors[i]]  # [n_neighbors, 3]
            diff = (neighbor_positions - vertices[i]) * 1e-3  # mm → m
            distances = np.linalg.norm(diff, axis=1)  # [n_neighbors]
            
            # dr = r_ij - r₀
            dr = distances - r0
            
            # U² = <dr²>
            U2[i] = np.mean(dr**2)
            
            # 体積ひずみ ≈ (V - V₀)/V₀ ≈ 3 × linear_strain
            # 簡易計算: Σdr/r₀ / n_neighbors
            volumetric_strain[i] = np.sum(dr) / (r0 * len(neighbors[i]))
        
        return U2, volumetric_strain
    
    def compute_lambda(self, U2: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        λ = U² / U²_c を計算
        
        λ < 1: 安定
        λ ≥ 1: 不安定（運命分岐へ）
        """
        U2_c = self.compute_U2_critical(Z)
        return U2 / np.maximum(U2_c, 1e-30)
    
    def determine_fate(self,
                       lambda_val: np.ndarray,
                       volumetric_strain: np.ndarray,
                       T: np.ndarray,
                       SPM: float = 20.0) -> np.ndarray:
        """
        λ > 1 後の運命を判定
        
        Args:
            lambda_val: λ値 [N]
            volumetric_strain: 体積ひずみ [N] (正=引張, 負=圧縮)
            T: 温度 [N] (K)
            SPM: ストローク/分
        
        Returns:
            fate: AtomFate の配列
        """
        n = len(lambda_val)
        fate = np.full(n, AtomFate.STABLE, dtype=object)
        
        # 閾値
        tension_threshold = 0.05      # 引張判定
        compression_threshold = -0.05  # 圧縮判定
        T_white_layer = self.mat.T_melt * 0.6  # 白層形成温度
        
        # 緩和可能かどうか（SPM依存）
        # 高SPM → 緩和不足 → 引張でCRACKしやすい
        relaxation_factor = np.clip(30.0 / SPM, 0.5, 2.0)  # SPM=30基準
        effective_tension_threshold = tension_threshold * relaxation_factor
        
        # 不安定な原子を抽出
        unstable = lambda_val >= 1.0
        
        for i in np.where(unstable)[0]:
            vol = volumetric_strain[i]
            temp = T[i] if hasattr(T, '__len__') else T
            
            if vol > effective_tension_threshold:
                # 引張 → CRACK（崖から落ちる）
                fate[i] = AtomFate.CRACK
            elif vol < compression_threshold:
                # 圧縮
                if temp > T_white_layer:
                    # 高温圧縮 → WHITE_LAYER
                    fate[i] = AtomFate.WHITE_LAYER
                else:
                    # 低温圧縮 → PLASTIC
                    fate[i] = AtomFate.PLASTIC
            else:
                # 剪断主体 → PLASTIC（隣の席に座る）
                fate[i] = AtomFate.PLASTIC
        
        return fate
    
    def analyze_mesh(self,
                     vertices: np.ndarray,
                     Z: np.ndarray,
                     vertices_reference: np.ndarray = None,
                     T: float = 300.0,
                     SPM: float = 20.0,
                     neighbor_cutoff_mm: float = 5.0) -> dict:
        """
        メッシュを解析
        
        Args:
            vertices: 現在の頂点座標 [N, 3] (mm)
            Z: 配位数 [N]
            vertices_reference: 参照座標（Noneなら近傍から推定）
            T: 温度 (K)
            SPM: ストローク/分
            neighbor_cutoff_mm: 近傍探索距離 (mm)
        
        Returns:
            解析結果のdict
        """
        n_vertices = len(vertices)
        
        # 1. 近傍リストを構築
        print("Building neighbor list...")
        tree = cKDTree(vertices)
        neighbors = tree.query_ball_point(vertices, neighbor_cutoff_mm)
        # 自分自身を除外
        neighbors = [np.array([j for j in nb if j != i]) for i, nb in enumerate(neighbors)]
        
        # 2. U²と体積ひずみを計算
        print("Computing U² and volumetric strain...")
        if vertices_reference is not None:
            # 参照座標があれば直接計算
            U2 = self.compute_U2_from_displacement(vertices, vertices_reference)
            # 体積ひずみは近傍から
            _, volumetric_strain = self.compute_U2_from_neighbor_stretch(
                vertices, neighbors
            )
        else:
            # 近傍距離から推定
            U2, volumetric_strain = self.compute_U2_from_neighbor_stretch(
                vertices, neighbors
            )
        
        # 3. λを計算
        print("Computing λ...")
        lambda_val = self.compute_lambda(U2, Z)
        
        # 4. 運命を判定
        print("Determining fate...")
        T_array = np.full(n_vertices, T)
        fate = self.determine_fate(lambda_val, volumetric_strain, T_array, SPM)
        
        # 5. 統計
        fate_counts = {f: np.sum(fate == f) for f in AtomFate}
        
        print(f"\n=== Analysis Results (SPM={SPM}) ===")
        print(f"Vertices: {n_vertices}")
        print(f"λ_max: {lambda_val.max():.3f}")
        print(f"λ_mean: {lambda_val.mean():.3f}")
        print(f"Volumetric strain range: [{volumetric_strain.min():.3f}, {volumetric_strain.max():.3f}]")
        print(f"\nFate distribution:")
        for f, count in fate_counts.items():
            pct = count / n_vertices * 100
            print(f"  {f.value:12s}: {count:6d} ({pct:5.1f}%)")
        
        # CRACK位置の詳細
        crack_mask = (fate == AtomFate.CRACK)
        if crack_mask.any():
            crack_verts = vertices[crack_mask]
            crack_Z = Z[crack_mask]
            print(f"\n⚠️ CRACK locations:")
            print(f"  Count: {crack_mask.sum()}")
            print(f"  X range: [{crack_verts[:,0].min():.1f}, {crack_verts[:,0].max():.1f}] mm")
            print(f"  Y range: [{crack_verts[:,1].min():.1f}, {crack_verts[:,1].max():.1f}] mm")
            print(f"  Z coord range: [{crack_verts[:,2].min():.1f}, {crack_verts[:,2].max():.1f}] mm")
            print(f"  Coordination Z: [{crack_Z.min()}, {crack_Z.max()}] (mean: {crack_Z.mean():.1f})")
        
        return {
            'U2': U2,
            'volumetric_strain': volumetric_strain,
            'lambda': lambda_val,
            'fate': fate,
            'fate_counts': fate_counts,
            'neighbors': neighbors,
        }


def simulate_press_step(vertices_before: np.ndarray,
                        vertices_after: np.ndarray,
                        Z: np.ndarray,
                        material: MaterialConstants,
                        SPM: float = 20.0,
                        T_ambient: float = 300.0) -> dict:
    """
    プレス1工程のシミュレーション
    
    Args:
        vertices_before: 変形前の座標 [N, 3] (mm)
        vertices_after: 変形後の座標 [N, 3] (mm)
        Z: 配位数 [N]
        material: 材料定数
        SPM: ストローク/分
        T_ambient: 環境温度 (K)
    
    Returns:
        解析結果
    """
    engine = U2PhysicsEngine(material)
    
    # 断熱加熱を推定
    # 高速変形 → 熱が逃げない → 温度上昇
    displacement = np.linalg.norm(vertices_after - vertices_before, axis=1)
    mean_disp = displacement.mean()
    
    # 簡易的な温度上昇モデル
    # ΔT ∝ (変形量) × (速度) / (熱容量)
    strain_rate_factor = SPM / 20.0  # SPM=20基準
    dT = 50.0 * (mean_disp / 10.0) * strain_rate_factor  # 最大50K程度
    T_local = T_ambient + dT
    
    print(f"Estimated temperature rise: ΔT = {dT:.1f} K → T = {T_local:.1f} K")
    
    # U²計算（変位から直接）
    U2 = engine.compute_U2_from_displacement(vertices_after, vertices_before)
    
    # 体積ひずみは近傍から
    tree = cKDTree(vertices_after)
    neighbors = tree.query_ball_point(vertices_after, 5.0)
    neighbors = [np.array([j for j in nb if j != i]) for i, nb in enumerate(neighbors)]
    _, volumetric_strain = engine.compute_U2_from_neighbor_stretch(vertices_after, neighbors)
    
    # λを計算
    lambda_val = engine.compute_lambda(U2, Z)
    
    # 運命判定
    T_array = np.full(len(vertices_after), T_local)
    fate = engine.determine_fate(lambda_val, volumetric_strain, T_array, SPM)
    
    fate_counts = {f: np.sum(fate == f) for f in AtomFate}
    
    return {
        'U2': U2,
        'volumetric_strain': volumetric_strain,
        'lambda': lambda_val,
        'fate': fate,
        'fate_counts': fate_counts,
        'T_local': T_local,
    }


# ========== テスト ==========
if __name__ == "__main__":
    # SECD材料でテスト
    mat = MaterialConstants.SECD()
    engine = U2PhysicsEngine(mat)
    
    print("\n" + "="*50)
    print("Test: U²_c vs Z")
    print("="*50)
    
    Z_values = np.array([3, 4, 5, 6, 7, 8])
    U2_c = engine.compute_U2_critical(Z_values)
    
    for z, u2c in zip(Z_values, U2_c):
        ratio = u2c / engine.U2_c_bulk
        print(f"Z={z}: U²_c = {u2c:.3e} m² ({ratio:.2f}x bulk)")
    
    print("\n" + "="*50)
    print("Test: Fate determination")
    print("="*50)
    
    # テストケース
    test_cases = [
        {"lambda": 0.5, "vol_strain": 0.0, "T": 300, "expected": "STABLE"},
        {"lambda": 1.5, "vol_strain": 0.1, "T": 300, "expected": "CRACK"},
        {"lambda": 1.5, "vol_strain": -0.1, "T": 300, "expected": "PLASTIC"},
        {"lambda": 1.5, "vol_strain": -0.1, "T": 1200, "expected": "WHITE_LAYER"},
        {"lambda": 1.5, "vol_strain": 0.0, "T": 300, "expected": "PLASTIC"},
    ]
    
    for tc in test_cases:
        fate = engine.determine_fate(
            np.array([tc["lambda"]]),
            np.array([tc["vol_strain"]]),
            np.array([tc["T"]]),
            SPM=20.0
        )[0]
        status = "✓" if fate.value == tc["expected"] else "✗"
        print(f"{status} λ={tc['lambda']}, vol={tc['vol_strain']:+.1f}, T={tc['T']}K → {fate.value} (expected: {tc['expected']})")
