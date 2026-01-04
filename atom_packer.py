"""
Atom Packing Module - メッシュ内部への原子充填
==============================================

メッシュは「人間用の形状グリッド」
その中に実際の原子を格子状に充填する

原子間距離からU²を計算するのが本質！
"""

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from dataclasses import dataclass
from typing import Tuple, Optional
import time


@dataclass
class CrystalStructure:
    """結晶構造"""
    name: str
    a: float           # 格子定数 [Å]
    basis: np.ndarray  # 単位胞内の原子位置（格子定数単位）
    Z_bulk: int        # バルク配位数
    r0: float          # 最近接原子間距離 [Å]
    
    @classmethod
    def BCC(cls, a: float = 2.87):
        """BCC構造（Fe, W, Cr...）"""
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        r0 = a * np.sqrt(3) / 2  # BCC最近接距離
        return cls(name="BCC", a=a, basis=basis, Z_bulk=8, r0=r0)
    
    @classmethod
    def FCC(cls, a: float = 3.61):
        """FCC構造（Al, Cu, Ni...）"""
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])
        r0 = a / np.sqrt(2)  # FCC最近接距離
        return cls(name="FCC", a=a, basis=basis, Z_bulk=12, r0=r0)


class AtomPacker:
    """
    メッシュ内部に原子を充填するクラス
    """
    
    def __init__(self, crystal: CrystalStructure):
        """
        Args:
            crystal: 結晶構造
        """
        self.crystal = crystal
        self.a_angstrom = crystal.a  # Å
        self.a_mm = crystal.a * 1e-7  # Å → mm (1Å = 10^-10 m = 10^-7 mm)
    
    def fill_box(self, 
                 box_min: np.ndarray, 
                 box_max: np.ndarray,
                 scale_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        直方体領域に原子を充填
        
        Args:
            box_min: 最小座標 [x, y, z] (mm)
            box_max: 最大座標 [x, y, z] (mm)
            scale_factor: スケール係数（シミュレーション用に拡大）
        
        Returns:
            positions: 原子位置 [N, 3] (mm)
            cell_indices: 単位胞インデックス [N, 3]
        """
        # 実際のスケール（1Å = 10^-7 mm）だと原子数が爆発するので
        # シミュレーション用にスケールアップ
        a_sim = self.a_mm * scale_factor
        
        # 各方向の単位胞数
        size = box_max - box_min
        n_cells = np.ceil(size / a_sim).astype(int)
        
        print(f"Filling box: {size} mm")
        print(f"Lattice constant (sim): {a_sim*1e7:.2f} Å (scale={scale_factor:.0e})")
        print(f"Unit cells: {n_cells} = {np.prod(n_cells)} cells")
        print(f"Atoms per cell: {len(self.crystal.basis)}")
        print(f"Expected atoms: {np.prod(n_cells) * len(self.crystal.basis)}")
        
        # 全原子位置を生成
        positions = []
        cell_indices = []
        
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    cell_origin = box_min + np.array([ix, iy, iz]) * a_sim
                    for basis_atom in self.crystal.basis:
                        pos = cell_origin + basis_atom * a_sim
                        positions.append(pos)
                        cell_indices.append([ix, iy, iz])
        
        return np.array(positions), np.array(cell_indices)
    
    def fill_mesh(self,
                  vertices: np.ndarray,
                  faces: np.ndarray,
                  thickness_mm: float = 1.96,
                  scale_factor: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
        """
        メッシュ内部に原子を充填
        
        Args:
            vertices: メッシュ頂点 [V, 3] (mm)
            faces: 面インデックス [F, 3]
            thickness_mm: 板厚 (mm)
            scale_factor: スケール係数（原子間距離を拡大）
        
        Returns:
            atom_positions: 原子位置 [N, 3] (mm)
            atom_Z: 配位数 [N]
        """
        # 1. バウンディングボックスで候補原子を生成
        box_min = vertices.min(axis=0) - thickness_mm
        box_max = vertices.max(axis=0) + thickness_mm
        
        print("="*60)
        print("Atom Packing into Mesh")
        print("="*60)
        
        candidates, _ = self.fill_box(box_min, box_max, scale_factor)
        print(f"Candidate atoms: {len(candidates)}")
        
        # 2. メッシュ内部の原子を抽出
        # 表面からの距離で判定（板厚を考慮）
        print("Filtering atoms inside mesh...")
        
        # メッシュ表面へのKD木
        tree = cKDTree(vertices)
        
        # 各候補原子について、最近傍表面点との距離を計算
        distances, _ = tree.query(candidates, k=1)
        
        # 板厚の半分以内にある原子を採用
        inside_mask = distances <= thickness_mm / 2
        atom_positions = candidates[inside_mask]
        
        print(f"Atoms inside mesh: {len(atom_positions)}")
        
        # 3. 配位数Zを計算
        print("Computing coordination numbers...")
        atom_Z = self._compute_coordination(atom_positions, scale_factor)
        
        return atom_positions, atom_Z
    
    def _compute_coordination(self, 
                               positions: np.ndarray,
                               scale_factor: float) -> np.ndarray:
        """
        配位数を計算
        
        Args:
            positions: 原子位置 [N, 3] (mm)
            scale_factor: スケール係数
        
        Returns:
            Z: 配位数 [N]
        """
        # 最近接距離（スケール済み）
        r_cutoff = self.crystal.r0 * 1e-7 * scale_factor * 1.2  # 20%マージン
        
        tree = cKDTree(positions)
        neighbors = tree.query_ball_point(positions, r_cutoff)
        
        # 自分自身を除いた近傍数
        Z = np.array([len(nb) - 1 for nb in neighbors])
        
        return Z
    
    def compute_U2(self,
                   positions: np.ndarray,
                   positions_reference: np.ndarray,
                   scale_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        U²（原子変位の二乗）と三軸度を計算
        
        Args:
            positions: 現在の原子位置 [N, 3] (mm)
            positions_reference: 参照位置 [N, 3] (mm)
            scale_factor: スケール係数
        
        Returns:
            U2: 変位の二乗 [N] (Å²単位に変換)
            volumetric_strain: 体積ひずみ [N]
            neighbor_list: 近傍リスト
        """
        n_atoms = len(positions)
        
        # 1. 変位ベクトル
        displacement = positions - positions_reference  # mm
        displacement_angstrom = displacement * 1e7 / scale_factor  # Å
        
        # U² = |displacement|²
        U2 = np.sum(displacement_angstrom**2, axis=1)
        
        # 2. 近傍リストを構築（現在位置）
        r_cutoff = self.crystal.r0 * 1e-7 * scale_factor * 1.5
        tree = cKDTree(positions)
        neighbors = tree.query_ball_point(positions, r_cutoff)
        
        # 3. 体積ひずみ（三軸度の指標）
        # 各原子について、近傍との距離変化から計算
        volumetric_strain = np.zeros(n_atoms)
        r0_mm = self.crystal.r0 * 1e-7 * scale_factor
        
        for i in range(n_atoms):
            nb_idx = [j for j in neighbors[i] if j != i]
            if len(nb_idx) == 0:
                continue
            
            # 現在の近傍距離
            r_current = np.linalg.norm(positions[nb_idx] - positions[i], axis=1)
            # 参照の近傍距離
            r_ref = np.linalg.norm(positions_reference[nb_idx] - positions_reference[i], axis=1)
            
            # 距離変化率の平均 ≈ 体積ひずみ / 3
            dr_over_r = (r_current - r_ref) / np.maximum(r_ref, 1e-10)
            volumetric_strain[i] = np.mean(dr_over_r) * 3
        
        return U2, volumetric_strain, neighbors


def pack_and_analyze(mesh_vertices: np.ndarray,
                     mesh_faces: np.ndarray,
                     mesh_vertices_ref: np.ndarray = None,
                     thickness_mm: float = 1.96,
                     T: float = 300.0,
                     SPM: float = 20.0,
                     scale_factor: float = 1e6) -> dict:
    """
    メッシュに原子を充填して解析
    
    Args:
        mesh_vertices: 現在のメッシュ頂点 (mm)
        mesh_faces: 面インデックス
        mesh_vertices_ref: 参照メッシュ頂点（Noneなら変形なし）
        thickness_mm: 板厚 (mm)
        T: 温度 (K)
        SPM: ストローク/分
        scale_factor: スケール係数
    
    Returns:
        解析結果
    """
    # SECD (BCC鉄)
    crystal = CrystalStructure.BCC(a=2.87)  # Å
    packer = AtomPacker(crystal)
    
    # 原子充填
    print(f"\n{'='*60}")
    print(f"ATOM PACKING ANALYSIS")
    print(f"Material: {crystal.name}, a = {crystal.a} Å")
    print(f"Temperature: {T} K, SPM: {SPM}")
    print(f"{'='*60}")
    
    atom_pos, atom_Z = packer.fill_mesh(
        mesh_vertices, mesh_faces, thickness_mm, scale_factor
    )
    
    if len(atom_pos) == 0:
        print("WARNING: No atoms inside mesh!")
        return None
    
    # 参照位置（変形前）
    if mesh_vertices_ref is None:
        # 参照なし → 変形量は近傍距離から推定
        atom_pos_ref = atom_pos.copy()
    else:
        # 参照あり → 対応する位置を補間で取得
        # 簡易実装: 最近傍の変位を適用
        tree_ref = cKDTree(mesh_vertices_ref)
        tree_cur = cKDTree(mesh_vertices)
        
        # 各原子について、最近傍メッシュ頂点の変位を取得
        _, idx_cur = tree_cur.query(atom_pos, k=1)
        mesh_displacement = mesh_vertices - mesh_vertices_ref
        atom_displacement = mesh_displacement[idx_cur]
        atom_pos_ref = atom_pos - atom_displacement
    
    # U² と体積ひずみを計算
    U2, vol_strain, neighbors = packer.compute_U2(
        atom_pos, atom_pos_ref, scale_factor
    )
    
    # 臨界U²（Lindemann基準）
    delta_L = 0.18  # SECD
    U2_c_bulk = (delta_L * crystal.a) ** 2  # Å²
    Z_bulk = crystal.Z_bulk
    
    # Z³スケーリング
    Z_ratio = np.clip(atom_Z / Z_bulk, 0.1, 2.0)
    U2_c = U2_c_bulk * (Z_ratio ** 3)
    
    # λ計算
    lambda_val = U2 / np.maximum(U2_c, 1e-10)
    
    # 運命判定
    from u2_physics_engine import AtomFate
    
    fate = np.full(len(atom_pos), AtomFate.STABLE, dtype=object)
    
    # 閾値
    tension_th = 0.05 * (30.0 / SPM)  # SPM依存
    compression_th = -0.05
    T_white = 1811 * 0.6  # Feの融点の60%
    
    unstable = lambda_val >= 1.0
    
    for i in np.where(unstable)[0]:
        vol = vol_strain[i]
        if vol > tension_th:
            fate[i] = AtomFate.CRACK
        elif vol < compression_th and T > T_white:
            fate[i] = AtomFate.WHITE_LAYER
        else:
            fate[i] = AtomFate.PLASTIC
    
    # 統計
    fate_counts = {f: np.sum(fate == f) for f in AtomFate}
    n_atoms = len(atom_pos)
    
    print(f"\n=== Results ===")
    print(f"Total atoms: {n_atoms}")
    print(f"λ_max: {lambda_val.max():.3f}")
    print(f"λ_mean: {lambda_val.mean():.3f}")
    print(f"U²_max: {U2.max():.3f} Å²")
    print(f"Vol strain range: [{vol_strain.min():.3f}, {vol_strain.max():.3f}]")
    print(f"\nFate distribution:")
    for f, count in fate_counts.items():
        print(f"  {f.value:12s}: {count:6d} ({count/n_atoms*100:5.1f}%)")
    
    return {
        'atom_positions': atom_pos,
        'atom_Z': atom_Z,
        'U2': U2,
        'volumetric_strain': vol_strain,
        'lambda': lambda_val,
        'fate': fate,
        'fate_counts': fate_counts,
        'crystal': crystal,
    }


# ========== テスト ==========
if __name__ == "__main__":
    # 単純なテスト：立方体に充填
    print("="*60)
    print("TEST: Fill cubic box")
    print("="*60)
    
    crystal = CrystalStructure.BCC(a=2.87)
    packer = AtomPacker(crystal)
    
    # 10mm x 10mm x 2mm の板
    box_min = np.array([0, 0, 0])
    box_max = np.array([10, 10, 2])
    
    # スケール 10^6（実際の1μm = シミュレーションの1mm）
    positions, indices = packer.fill_box(box_min, box_max, scale_factor=1e6)
    
    print(f"\nGenerated {len(positions)} atoms")
    print(f"Position range: {positions.min(axis=0)} to {positions.max(axis=0)}")
    
    # 配位数
    Z = packer._compute_coordination(positions, scale_factor=1e6)
    print(f"Z range: {Z.min()} to {Z.max()} (mean: {Z.mean():.1f})")
    print(f"Bulk atoms (Z={crystal.Z_bulk}): {(Z == crystal.Z_bulk).sum()}")
    print(f"Surface atoms (Z<{crystal.Z_bulk}): {(Z < crystal.Z_bulk).sum()}")
