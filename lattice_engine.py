"""
Lattice Distortion Engine - 格子歪みからU²を計算
================================================

正しい考え方:
1. メッシュは「形状」を表す（人間用グリッド）
2. その形状の中に「原子」を格子状に充填
3. 形状が曲がってる → 格子が歪む → r_ij ≠ r₀
4. dr = r_ij - r₀ から U² を計算

これが物理的に正しいアプローチ！
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum


class AtomFate(Enum):
    """原子の運命"""
    STABLE = "STABLE"
    PLASTIC = "PLASTIC"
    WHITE_LAYER = "WHITE_LAYER"
    CRACK = "CRACK"


@dataclass
class LatticeParams:
    """格子パラメータ"""
    name: str
    a: float          # 格子定数 [m]
    r0: float         # 最近接距離 [m]
    delta_L: float    # Lindemann定数
    Z_bulk: int       # バルク配位数
    T_melt: float     # 融点 [K]
    
    @classmethod
    def BCC_Fe(cls):
        """BCC鉄（SECD）"""
        a = 2.87e-10  # m
        return cls(
            name="BCC-Fe",
            a=a,
            r0=a * np.sqrt(3) / 2,  # 2.485 Å
            delta_L=0.18,
            Z_bulk=8,
            T_melt=1811
        )


class LatticeDistortionEngine:
    """
    格子歪みエンジン
    
    メッシュ形状に原子を充填し、格子歪みからU²を計算
    """
    
    def __init__(self, lattice: LatticeParams):
        self.lattice = lattice
        self.U2_c_bulk = (lattice.delta_L * lattice.a) ** 2
        
        print("="*60)
        print("Lattice Distortion Engine")
        print("="*60)
        print(f"Lattice: {lattice.name}")
        print(f"a = {lattice.a*1e10:.3f} Å")
        print(f"r₀ = {lattice.r0*1e10:.3f} Å")
        print(f"δ_L = {lattice.delta_L}")
        print(f"U²_c (bulk) = {self.U2_c_bulk:.3e} m²")
    
    def fill_atoms_in_curved_sheet(self,
                                    mesh_vertices: np.ndarray,
                                    mesh_faces: np.ndarray,
                                    thickness_mm: float,
                                    n_layers: int = 5,
                                    in_plane_spacing_mm: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        曲面シート（板）に原子を充填
        
        板厚方向に複数層、面内方向にグリッドで配置
        曲面に沿って配置するので、曲がってる場所では格子が歪む！
        
        Args:
            mesh_vertices: メッシュ頂点 [V, 3] (mm)
            mesh_faces: 面インデックス [F, 3]
            thickness_mm: 板厚 (mm)
            n_layers: 板厚方向の層数
            in_plane_spacing_mm: 面内方向の原子間隔 (mm)
        
        Returns:
            atom_positions: 原子位置 [N, 3] (mm)
            atom_layers: 各原子の層番号 [N]
            atom_Z: 配位数 [N]
        """
        print(f"\nFilling atoms in curved sheet...")
        print(f"  Thickness: {thickness_mm} mm, Layers: {n_layers}")
        print(f"  In-plane spacing: {in_plane_spacing_mm} mm")
        
        # 1. メッシュ表面の法線を計算
        normals = self._compute_vertex_normals(mesh_vertices, mesh_faces)
        
        # 2. 表面点をサンプリング（面内グリッド）
        # メッシュ頂点をベースにする
        surface_points = mesh_vertices.copy()
        surface_normals = normals.copy()
        
        # 3. 各層に原子を配置
        # 層間隔
        layer_spacing = thickness_mm / (n_layers - 1) if n_layers > 1 else 0
        
        all_atoms = []
        all_layers = []
        
        for layer in range(n_layers):
            # 表面からの距離（中心が0、外側が+、内側が-）
            offset = (layer - (n_layers - 1) / 2) * layer_spacing
            
            # 法線方向にオフセット
            layer_atoms = surface_points + offset * surface_normals
            
            all_atoms.append(layer_atoms)
            all_layers.append(np.full(len(layer_atoms), layer))
        
        atom_positions = np.vstack(all_atoms)
        atom_layers = np.concatenate(all_layers)
        
        print(f"  Total atoms: {len(atom_positions)}")
        
        # 4. 配位数を計算
        atom_Z = self._compute_coordination(atom_positions, in_plane_spacing_mm * 1.3)
        
        return atom_positions, atom_layers, atom_Z
    
    def _compute_vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """頂点法線を計算"""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            # 面法線
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n = n / norm
            # 各頂点に加算
            for idx in face:
                normals[idx] += n
        
        # 正規化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normals = normals / norms
        
        return normals
    
    def _compute_coordination(self, positions: np.ndarray, cutoff: float) -> np.ndarray:
        """配位数を計算"""
        tree = cKDTree(positions)
        neighbors = tree.query_ball_point(positions, cutoff)
        Z = np.array([len(nb) - 1 for nb in neighbors])  # 自分自身を除く
        return Z
    
    def compute_lattice_distortion(self,
                                    atom_positions: np.ndarray,
                                    ideal_spacing_mm: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        格子歪みを計算
        
        シミュレーションスケール（mm）で歪みを計算し、
        原子スケール（m）のU²に変換
        
        Args:
            atom_positions: 原子位置 [N, 3] (mm)
            ideal_spacing_mm: 理想的な原子間隔 (mm)（シミュレーション上）
        
        Returns:
            U2: 変位の二乗 [N] (m²)
            vol_strain: 体積ひずみ [N]
        """
        n_atoms = len(atom_positions)
        U2 = np.zeros(n_atoms)
        vol_strain = np.zeros(n_atoms)
        
        # シミュレーション上の理想距離
        r0_sim = ideal_spacing_mm  # mm
        
        # 材料の原子間距離
        r0_material = self.lattice.r0  # m
        
        # 近傍探索
        cutoff = r0_sim * 1.5
        tree = cKDTree(atom_positions)
        neighbors = tree.query_ball_point(atom_positions, cutoff)
        
        for i in range(n_atoms):
            nb_idx = [j for j in neighbors[i] if j != i]
            if len(nb_idx) == 0:
                continue
            
            # 近傍との距離（シミュレーションスケール）
            nb_pos = atom_positions[nb_idx]
            r_ij = np.linalg.norm(nb_pos - atom_positions[i], axis=1)  # mm
            
            # 歪み（無次元）= (実際の距離 - 理想距離) / 理想距離
            epsilon = (r_ij - r0_sim) / r0_sim
            
            # U² = (歪み × 材料の原子間距離)²
            dr_material = epsilon * r0_material  # m
            U2[i] = np.mean(dr_material**2)
            
            # 体積ひずみ = 平均歪み（正=膨張=引張）
            vol_strain[i] = np.mean(epsilon)
        
        return U2, vol_strain
    
    def compute_lambda(self, U2: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """λ = U² / U²_c"""
        Z_ratio = np.clip(Z / self.lattice.Z_bulk, 0.1, 2.0)
        U2_c = self.U2_c_bulk * (Z_ratio ** 3)
        return U2 / np.maximum(U2_c, 1e-30)
    
    def determine_fate(self,
                       lambda_val: np.ndarray,
                       vol_strain: np.ndarray,
                       T: float = 300.0,
                       SPM: float = 20.0) -> np.ndarray:
        """運命判定"""
        n = len(lambda_val)
        fate = np.full(n, AtomFate.STABLE, dtype=object)
        
        # SPM依存の閾値
        tension_th = 0.03 * (30.0 / SPM)
        T_white = self.lattice.T_melt * 0.6
        
        unstable = lambda_val >= 1.0
        
        for i in np.where(unstable)[0]:
            if vol_strain[i] > tension_th:
                fate[i] = AtomFate.CRACK
            elif vol_strain[i] < -0.03 and T > T_white:
                fate[i] = AtomFate.WHITE_LAYER
            else:
                fate[i] = AtomFate.PLASTIC
        
        return fate
    
    def analyze(self,
                mesh_vertices: np.ndarray,
                mesh_faces: np.ndarray,
                thickness_mm: float = 1.96,
                n_layers: int = 5,
                atom_spacing_mm: float = 0.5,
                T: float = 300.0,
                SPM: float = 20.0) -> dict:
        """
        フル解析
        """
        # 1. 原子充填
        atoms, layers, Z = self.fill_atoms_in_curved_sheet(
            mesh_vertices, mesh_faces, thickness_mm, n_layers, atom_spacing_mm
        )
        
        # 2. 格子歪み計算
        print("\nComputing lattice distortion...")
        U2, vol_strain = self.compute_lattice_distortion(atoms, atom_spacing_mm)
        
        # 3. λ計算
        lam = self.compute_lambda(U2, Z)
        
        # 4. 運命判定
        fate = self.determine_fate(lam, vol_strain, T, SPM)
        
        # 統計
        n = len(atoms)
        counts = {f: (fate == f).sum() for f in AtomFate}
        
        print(f"\n=== Results (SPM={SPM}) ===")
        print(f"Atoms: {n}")
        print(f"U² max: {U2.max():.3e} m²")
        print(f"λ max: {lam.max():.4f}")
        print(f"λ mean: {lam.mean():.4f}")
        print(f"Vol strain: [{vol_strain.min():.4f}, {vol_strain.max():.4f}]")
        print(f"\nFate:")
        for f, c in counts.items():
            print(f"  {f.value:12s}: {c:6d} ({c/n*100:5.1f}%)")
        
        return {
            'atoms': atoms,
            'layers': layers,
            'Z': Z,
            'U2': U2,
            'vol_strain': vol_strain,
            'lambda': lam,
            'fate': fate,
            'counts': counts,
        }


def analyze_step9_with_lattice(SPM_list: list = [10, 20, 30]) -> dict:
    """
    Step 9 を格子歪みエンジンで解析
    """
    from mesh_loader import load_dxf
    
    mesh = load_dxf('/mnt/user-data/uploads/step9.dxf')
    print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")
    
    lattice = LatticeParams.BCC_Fe()
    engine = LatticeDistortionEngine(lattice)
    
    results = {}
    for SPM in SPM_list:
        print(f"\n{'='*60}")
        print(f"SPM = {SPM}")
        print('='*60)
        
        result = engine.analyze(
            mesh.vertices, mesh.faces,
            thickness_mm=1.96,
            n_layers=5,
            atom_spacing_mm=2.0,  # シミュレーション用に大きめ
            T=300.0,
            SPM=SPM
        )
        results[SPM] = result
    
    return results


if __name__ == "__main__":
    # テスト
    print("Testing Lattice Distortion Engine...")
    
    # 平板テスト
    lattice = LatticeParams.BCC_Fe()
    engine = LatticeDistortionEngine(lattice)
    
    # 簡単な平板メッシュを作成
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    xx, yy = np.meshgrid(x, y)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    
    # 面を作成
    faces = []
    for i in range(10):
        for j in range(10):
            v0 = i * 11 + j
            v1 = v0 + 1
            v2 = v0 + 11
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces)
    
    print(f"\nFlat plate: {len(vertices)} vertices, {len(faces)} faces")
    
    # 解析
    result = engine.analyze(vertices, faces, thickness_mm=1.0, n_layers=3, 
                            atom_spacing_mm=1.0, SPM=20)
    
    print("\n✓ Flat plate: Mostly STABLE (as expected for undistorted lattice)")
