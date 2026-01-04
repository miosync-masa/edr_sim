"""
Geometry Classification Engine
==============================

CAD形状からサーフェス/エッジ/コーナーを分類し、
物理的に正しいZ_eff（有効配位数）を割り当てる

分類:
  SURFACE（面）   : Z_eff ≈ 6-7 (片面欠損)
  EDGE（稜線）    : Z_eff ≈ 4-5 (二面交差)
  CORNER（角）    : Z_eff ≈ 3   (三面以上交差)
  BULK（内部）    : Z_eff = 8   (完全配位)

板の場合、ほぼ全部SURFACE（表裏）だが、
R部やベアリングポケットはEDGE/CORNERになる
"""

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum


class GeometryType(Enum):
    """ジオメトリ分類"""
    BULK = "BULK"         # 内部（板では存在しない）
    SURFACE = "SURFACE"   # 面
    EDGE = "EDGE"         # 稜線
    CORNER = "CORNER"     # 角


@dataclass
class GeometryClassification:
    """ジオメトリ分類結果"""
    types: np.ndarray           # GeometryType配列
    Z_eff: np.ndarray           # 有効配位数
    normal_variance: np.ndarray  # 法線分散（診断用）
    curvature: np.ndarray       # 曲率（診断用）


class GeometryClassifier:
    """
    CAD形状からジオメトリ分類を行うクラス
    """
    
    # BCC鉄の配位数
    Z_BULK = 8
    Z_SURFACE = 6  # 片面欠損
    Z_EDGE = 4     # 二面交差
    Z_CORNER = 3   # 三面交差
    
    def __init__(self, 
                 edge_angle_threshold: float = 30.0,
                 corner_angle_threshold: float = 60.0):
        """
        Args:
            edge_angle_threshold: エッジ判定の角度閾値 [度]
            corner_angle_threshold: コーナー判定の角度閾値 [度]
        """
        self.edge_angle_th = np.radians(edge_angle_threshold)
        self.corner_angle_th = np.radians(corner_angle_threshold)
    
    def classify(self, 
                 vertices: np.ndarray,
                 faces: np.ndarray) -> GeometryClassification:
        """
        メッシュを分類
        
        Args:
            vertices: 頂点座標 [V, 3]
            faces: 面インデックス [F, 3]
        
        Returns:
            分類結果
        """
        n_vertices = len(vertices)
        
        print("="*60)
        print("Geometry Classification")
        print("="*60)
        print(f"Vertices: {n_vertices}")
        
        # 1. 面法線を計算
        print("\nComputing face normals...")
        face_normals = self._compute_face_normals(vertices, faces)
        
        # 2. 頂点ごとの隣接面法線を収集
        print("Computing vertex normal variance...")
        vertex_face_map = self._build_vertex_face_map(vertices, faces)
        
        # 3. 法線分散を計算
        normal_variance = np.zeros(n_vertices)
        for i in range(n_vertices):
            adj_faces = vertex_face_map[i]
            if len(adj_faces) < 2:
                normal_variance[i] = 0
                continue
            
            normals = face_normals[list(adj_faces)]
            # 法線間の最大角度差を計算
            max_angle = 0
            for j in range(len(normals)):
                for k in range(j+1, len(normals)):
                    dot = np.clip(np.dot(normals[j], normals[k]), -1, 1)
                    angle = np.arccos(dot)
                    max_angle = max(max_angle, angle)
            normal_variance[i] = max_angle
        
        # 4. 曲率も計算（補助情報）
        print("Computing curvature...")
        curvature = self._compute_curvature(vertices, faces)
        
        # 5. 分類
        print("Classifying geometry...")
        types = np.full(n_vertices, GeometryType.SURFACE, dtype=object)
        Z_eff = np.full(n_vertices, self.Z_SURFACE, dtype=float)
        
        for i in range(n_vertices):
            if normal_variance[i] > self.corner_angle_th:
                types[i] = GeometryType.CORNER
                Z_eff[i] = self.Z_CORNER
            elif normal_variance[i] > self.edge_angle_th:
                types[i] = GeometryType.EDGE
                Z_eff[i] = self.Z_EDGE
            else:
                types[i] = GeometryType.SURFACE
                Z_eff[i] = self.Z_SURFACE
        
        # 統計
        type_counts = {t: (types == t).sum() for t in GeometryType}
        print(f"\nClassification results:")
        for t, count in type_counts.items():
            pct = count / n_vertices * 100
            print(f"  {t.value:8s}: {count:6d} ({pct:5.1f}%)")
        
        print(f"\nZ_eff statistics:")
        print(f"  min: {Z_eff.min():.1f}")
        print(f"  max: {Z_eff.max():.1f}")
        print(f"  mean: {Z_eff.mean():.2f}")
        
        return GeometryClassification(
            types=types,
            Z_eff=Z_eff,
            normal_variance=normal_variance,
            curvature=curvature
        )
    
    def _compute_face_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """面法線を計算"""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-10)
        
        return normals
    
    def _build_vertex_face_map(self, vertices: np.ndarray, faces: np.ndarray) -> list:
        """各頂点の隣接面を構築"""
        n_vertices = len(vertices)
        vertex_face_map = [set() for _ in range(n_vertices)]
        
        for fi, face in enumerate(faces):
            for vi in face:
                vertex_face_map[vi].add(fi)
        
        return vertex_face_map
    
    def _compute_curvature(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """曲率を計算（簡易版）"""
        n_vertices = len(vertices)
        curvature = np.zeros(n_vertices)
        
        # 隣接頂点を構築
        adjacency = [set() for _ in range(n_vertices)]
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adjacency[face[i]].add(face[j])
        
        # 頂点法線
        vertex_normals = np.zeros_like(vertices)
        face_normals = self._compute_face_normals(vertices, faces)
        for fi, face in enumerate(faces):
            for vi in face:
                vertex_normals[vi] += face_normals[fi]
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / np.maximum(norms, 1e-10)
        
        # 曲率 = 近傍点との高さ差 / 距離²
        for i in range(n_vertices):
            neighbors = list(adjacency[i])
            if len(neighbors) < 3:
                continue
            
            p_i = vertices[i]
            n_i = vertex_normals[i]
            
            neighbor_pos = vertices[neighbors]
            rel_pos = neighbor_pos - p_i
            
            # 法線方向の「高さ」
            heights = np.dot(rel_pos, n_i)
            
            # 接平面内の距離
            tangent_sq = np.sum(rel_pos**2, axis=1) - heights**2
            tangent_dist = np.sqrt(np.maximum(tangent_sq, 1e-10))
            
            # 曲率 ≈ 2h / r²
            local_curvatures = np.abs(2 * heights / (tangent_dist**2 + 1e-10))
            curvature[i] = np.mean(local_curvatures)
        
        return curvature


def analyze_with_geometry_classification(mesh_vertices: np.ndarray,
                                          mesh_faces: np.ndarray,
                                          thickness_mm: float = 1.96,
                                          T: float = 300.0,
                                          SPM: float = 20.0) -> dict:
    """
    ジオメトリ分類を使った完全解析
    """
    from delta_mech_analyzer import compute_vertex_curvature, AtomFate
    
    print("\n" + "="*70)
    print("FULL ANALYSIS: Geometry Classification + δ_mech")
    print("="*70)
    
    # 1. ジオメトリ分類
    classifier = GeometryClassifier(
        edge_angle_threshold=30.0,
        corner_angle_threshold=60.0
    )
    geo = classifier.classify(mesh_vertices, mesh_faces)
    
    # 2. 曲率を計算
    print("\nComputing bending curvature...")
    mean_curv, _ = compute_vertex_curvature(mesh_vertices, mesh_faces)
    
    # 3. δ_mech を計算（Z_eff使用）
    print("\nComputing δ_mech with Z_eff...")
    
    # 曲げひずみ
    epsilon_bend = thickness_mm * np.abs(mean_curv) / 2.0
    
    # Z効果: (Z_bulk / Z_eff) で増幅
    # 表面(Z=6): 8/6 = 1.33倍
    # エッジ(Z=4): 8/4 = 2倍
    # コーナー(Z=3): 8/3 = 2.67倍
    Z_bulk = 8
    Z_factor = Z_bulk / np.maximum(geo.Z_eff, 1)
    
    # δ_mech = ε × √Z_factor (√にして効果を緩める)
    delta_mech = epsilon_bend * np.sqrt(Z_factor)
    
    print(f"  δ_mech range: [{delta_mech.min():.4f}, {delta_mech.max():.4f}]")
    print(f"  δ_mech mean: {delta_mech.mean():.4f}")
    
    # 4. Lindemann判定
    delta_threshold = 0.10
    exceeds = delta_mech > delta_threshold
    print(f"  δ > {delta_threshold}: {exceeds.sum()} ({exceeds.sum()/len(delta_mech)*100:.2f}%)")
    
    # 5. 体積ひずみ（三軸度）
    vol_strain = mean_curv * thickness_mm / 2
    
    # 6. 運命判定
    print("\nDetermining fate...")
    n = len(mesh_vertices)
    fate = np.full(n, AtomFate.STABLE, dtype=object)
    
    # SPM依存閾値
    tension_th = 0.02 * (30.0 / SPM)
    
    for i in range(n):
        if delta_mech[i] < delta_threshold:
            fate[i] = AtomFate.STABLE
        elif vol_strain[i] > tension_th:
            fate[i] = AtomFate.CRACK
        else:
            fate[i] = AtomFate.PLASTIC
    
    counts = {f: (fate == f).sum() for f in AtomFate}
    print(f"\nFate distribution:")
    for f, c in counts.items():
        print(f"  {f.value:12s}: {c:6d} ({c/n*100:5.1f}%)")
    
    # CRACK位置の詳細
    crack_mask = (fate == AtomFate.CRACK)
    if crack_mask.sum() > 0:
        crack_pos = mesh_vertices[crack_mask]
        crack_geo = geo.types[crack_mask]
        crack_Z = geo.Z_eff[crack_mask]
        
        print(f"\n⚠️  CRACK locations:")
        print(f"  Count: {crack_mask.sum()}")
        print(f"  Position range:")
        print(f"    X: [{crack_pos[:,0].min():.1f}, {crack_pos[:,0].max():.1f}] mm")
        print(f"    Y: [{crack_pos[:,1].min():.1f}, {crack_pos[:,1].max():.1f}] mm")
        print(f"  Geometry breakdown:")
        for t in GeometryType:
            cnt = (crack_geo == t).sum()
            if cnt > 0:
                print(f"    {t.value}: {cnt} ({cnt/crack_mask.sum()*100:.1f}%)")
        print(f"  Z_eff range: [{crack_Z.min():.1f}, {crack_Z.max():.1f}]")
    
    return {
        'geometry': geo,
        'curvature': mean_curv,
        'delta_mech': delta_mech,
        'vol_strain': vol_strain,
        'fate': fate,
        'counts': counts,
    }


# ========================================
# メイン
# ========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mesh_loader import load_dxf
    from delta_mech_analyzer import AtomFate
    
    print("Loading Step 9 mesh...")
    mesh = load_dxf('/mnt/user-data/uploads/step9.dxf')
    
    # SPM別解析
    results = {}
    for SPM in [10, 20, 30]:
        print(f"\n{'#'*70}")
        print(f"# SPM = {SPM}")
        print('#'*70)
        
        result = analyze_with_geometry_classification(
            mesh.vertices, mesh.faces,
            thickness_mm=1.96,
            T=300.0,
            SPM=SPM
        )
        results[SPM] = result
    
    # 可視化
    fig = plt.figure(figsize=(20, 12))
    
    # 1. ジオメトリ分類マップ
    ax1 = fig.add_subplot(231, projection='3d')
    geo = results[20]['geometry']
    colors = np.array(['blue' if t == GeometryType.SURFACE else
                       'orange' if t == GeometryType.EDGE else
                       'red' for t in geo.types])
    n_sample = min(3000, len(mesh.vertices))
    idx = np.random.choice(len(mesh.vertices), n_sample, replace=False)
    ax1.scatter(mesh.vertices[idx,0], mesh.vertices[idx,1], mesh.vertices[idx,2],
                c=colors[idx], s=2, alpha=0.6)
    ax1.set_title('Geometry Classification\n(Blue=Surface, Orange=Edge, Red=Corner)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # 2. Z_eff分布
    ax2 = fig.add_subplot(232)
    ax2.hist(geo.Z_eff, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Z_eff')
    ax2.set_ylabel('Count')
    ax2.set_title('Z_eff Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. δ_mech分布
    ax3 = fig.add_subplot(233)
    delta = results[20]['delta_mech']
    ax3.hist(delta[delta < 0.3], bins=50, alpha=0.7, edgecolor='black', label='All')
    tail = delta[delta > 0.10]
    if len(tail) > 0:
        ax3.hist(tail[tail < 0.3], bins=50, alpha=0.9, color='red', 
                edgecolor='darkred', label=f'δ>0.1 ({len(tail)/len(delta)*100:.1f}%)')
    ax3.axvline(0.10, color='black', linestyle='--', linewidth=2, label='Lindemann')
    ax3.set_xlabel('δ_mech')
    ax3.set_ylabel('Count')
    ax3.set_title('δ_mech Distribution (SPM=20)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. SPM vs CRACK
    ax4 = fig.add_subplot(234)
    spms = [10, 20, 30]
    crack_pcts = [results[s]['counts'][AtomFate.CRACK]/len(mesh.vertices)*100 for s in spms]
    ax4.bar(spms, crack_pcts, color='red', alpha=0.7)
    ax4.set_xlabel('SPM')
    ax4.set_ylabel('CRACK [%]')
    ax4.set_title('CRACK Risk vs SPM')
    ax4.grid(True, alpha=0.3)
    
    # 5. Fateマップ (SPM=20)
    ax5 = fig.add_subplot(235, projection='3d')
    fate = results[20]['fate']
    colors_fate = np.array(['green' if f == AtomFate.STABLE else
                            'yellow' if f == AtomFate.PLASTIC else
                            'red' for f in fate])
    ax5.scatter(mesh.vertices[idx,0], mesh.vertices[idx,1], mesh.vertices[idx,2],
                c=colors_fate[idx], s=2, alpha=0.6)
    ax5.set_title('Fate Map (SPM=20)\n(Green=OK, Yellow=PLASTIC, Red=CRACK)')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y'); ax5.set_zlabel('Z')
    
    # 6. ジオメトリ別CRACK率
    ax6 = fig.add_subplot(236)
    fate_20 = results[20]['fate']
    geo_types = [GeometryType.SURFACE, GeometryType.EDGE, GeometryType.CORNER]
    crack_by_geo = []
    for t in geo_types:
        mask = geo.types == t
        if mask.sum() > 0:
            crack_rate = (fate_20[mask] == AtomFate.CRACK).sum() / mask.sum() * 100
        else:
            crack_rate = 0
        crack_by_geo.append(crack_rate)
    
    colors_geo = ['blue', 'orange', 'red']
    ax6.bar([t.value for t in geo_types], crack_by_geo, color=colors_geo, alpha=0.7)
    ax6.set_xlabel('Geometry Type')
    ax6.set_ylabel('CRACK Rate [%]')
    ax6.set_title('CRACK Risk by Geometry\n(Corner > Edge > Surface)')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Step 9: Geometry-Aware δ_mech Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/step9_geometry_aware.png', dpi=150)
    print(f"\nSaved: /mnt/user-data/outputs/step9_geometry_aware.png")
    
    # サマリ
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for SPM in spms:
        n = len(mesh.vertices)
        c = results[SPM]['counts']
        print(f"SPM={SPM}: STABLE {c[AtomFate.STABLE]/n*100:.1f}%, "
              f"PLASTIC {c[AtomFate.PLASTIC]/n*100:.1f}%, "
              f"CRACK {c[AtomFate.CRACK]/n*100:.1f}%")
