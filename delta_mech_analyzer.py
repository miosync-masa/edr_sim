"""
Curvature-Based δ Analysis
===========================

CAD形状の曲率から機械的Lindemann比 δ_mech を計算

物理:
  曲げひずみ: ε_bend = t / (2R)  (Rは曲率半径)
  δ_mech ∝ ε  (局所的な格子変位)
  
  δ_mech > 0.1 → 局所的なLindemann超過 → 危険！
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple
from enum import Enum


class AtomFate(Enum):
    STABLE = "STABLE"
    PLASTIC = "PLASTIC"  
    WHITE_LAYER = "WHITE_LAYER"
    CRACK = "CRACK"


def compute_vertex_curvature(vertices: np.ndarray, 
                              faces: np.ndarray,
                              neighbor_rings: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    各頂点の主曲率を計算
    
    Returns:
        mean_curvature: 平均曲率 H = (κ₁ + κ₂)/2
        gaussian_curvature: ガウス曲率 K = κ₁ × κ₂
    """
    n_vertices = len(vertices)
    mean_curvature = np.zeros(n_vertices)
    gaussian_curvature = np.zeros(n_vertices)
    
    # 隣接頂点を構築
    adjacency = [set() for _ in range(n_vertices)]
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[face[i]].add(face[j])
    
    # 法線を計算
    normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
        for idx in face:
            normals[idx] += n
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-10)
    
    # 各頂点の曲率を計算（離散微分幾何）
    for i in range(n_vertices):
        neighbors = list(adjacency[i])
        if len(neighbors) < 3:
            continue
        
        # 近傍点との距離と角度から曲率を推定
        p_i = vertices[i]
        n_i = normals[i]
        
        # 接平面への射影
        neighbor_pos = vertices[neighbors]
        rel_pos = neighbor_pos - p_i
        
        # 法線方向の距離（曲率に関係）
        heights = np.dot(rel_pos, n_i)
        
        # 接平面内の距離
        tangent_dist = np.sqrt(np.sum(rel_pos**2, axis=1) - heights**2 + 1e-10)
        
        # 曲率 ≈ 2h / r² (放物面近似)
        curvatures = 2 * heights / (tangent_dist**2 + 1e-10)
        
        # 平均と分散
        mean_curvature[i] = np.mean(curvatures)
        gaussian_curvature[i] = np.var(curvatures)  # 近似
    
    return mean_curvature, gaussian_curvature


def compute_delta_mech(curvature: np.ndarray,
                        thickness_mm: float,
                        Z: np.ndarray,
                        Z_bulk: int = 8) -> np.ndarray:
    """
    曲率から機械的Lindemann比 δ_mech を計算
    
    δ_mech = ε_bend × f(Z)
    
    ε_bend = t × |κ| / 2  (曲げひずみ)
    f(Z) = (Z_bulk / Z)  (配位数効果: 低Zで増幅)
    
    Args:
        curvature: 曲率 [1/mm]
        thickness_mm: 板厚 [mm]
        Z: 配位数
        Z_bulk: バルク配位数
    
    Returns:
        delta_mech: 機械的Lindemann比
    """
    # 曲げひずみ: ε = t × |κ| / 2
    # κ [1/mm] なので ε は無次元
    epsilon_bend = thickness_mm * np.abs(curvature) / 2.0
    
    # 配位数効果: 低Zで δ が増幅される
    Z_factor = Z_bulk / np.maximum(Z, 1)
    
    # δ_mech = ε × Z_factor
    # ε = 0.1 (10%) で δ ≈ 0.1 になるようにスケーリング
    delta_mech = epsilon_bend * Z_factor
    
    return delta_mech


def analyze_cad_shape(mesh_vertices: np.ndarray,
                       mesh_faces: np.ndarray,
                       mesh_Z: np.ndarray,
                       thickness_mm: float = 1.96,
                       T: float = 300.0,
                       SPM: float = 20.0) -> dict:
    """
    CAD形状から δ 分布を計算し、運命を判定
    
    Args:
        mesh_vertices: メッシュ頂点 [V, 3] (mm)
        mesh_faces: 面インデックス [F, 3]
        mesh_Z: 配位数 [V]
        thickness_mm: 板厚 (mm)
        T: 温度 (K)
        SPM: ストローク/分
    
    Returns:
        解析結果
    """
    print("="*60)
    print("CAD Shape → δ_mech Analysis")
    print("="*60)
    print(f"Vertices: {len(mesh_vertices)}")
    print(f"Thickness: {thickness_mm} mm")
    print(f"T: {T} K, SPM: {SPM}")
    
    # 1. 曲率を計算
    print("\nComputing curvature...")
    mean_curv, gauss_curv = compute_vertex_curvature(mesh_vertices, mesh_faces)
    
    print(f"  Mean curvature range: [{mean_curv.min():.4f}, {mean_curv.max():.4f}] 1/mm")
    print(f"  Curvature radius range: [{1/np.maximum(np.abs(mean_curv).max(), 1e-6):.1f}, ∞] mm")
    
    # 2. δ_mech を計算
    print("\nComputing δ_mech...")
    delta_mech = compute_delta_mech(mean_curv, thickness_mm, mesh_Z)
    
    print(f"  δ_mech range: [{delta_mech.min():.4f}, {delta_mech.max():.4f}]")
    print(f"  δ_mech mean: {delta_mech.mean():.4f}")
    
    # 3. Lindemann閾値との比較
    delta_threshold = 0.10  # Lindemann
    exceeds = delta_mech > delta_threshold
    
    print(f"\n  δ > {delta_threshold}: {exceeds.sum()} vertices ({exceeds.sum()/len(delta_mech)*100:.2f}%)")
    
    # 4. 体積ひずみ（三軸度）を推定
    # 曲げの外側: 引張 (δ > 0, curv > 0 と仮定)
    # 曲げの内側: 圧縮 (δ > 0, curv < 0 と仮定)
    # ここでは曲率の符号で判定
    vol_strain = mean_curv * thickness_mm / 2  # 簡易推定
    
    # 5. 運命判定
    print("\nDetermining fate...")
    
    fate = np.full(len(mesh_vertices), AtomFate.STABLE, dtype=object)
    
    # SPM依存の閾値
    tension_th = 0.02 * (30.0 / SPM)
    T_white = 1811 * 0.6  # Fe融点の60%
    
    # δ > threshold の頂点について判定
    for i in np.where(exceeds)[0]:
        if vol_strain[i] > tension_th:
            fate[i] = AtomFate.CRACK
        elif vol_strain[i] < -tension_th and T > T_white:
            fate[i] = AtomFate.WHITE_LAYER
        else:
            fate[i] = AtomFate.PLASTIC
    
    # 統計
    n = len(mesh_vertices)
    counts = {f: (fate == f).sum() for f in AtomFate}
    
    print(f"\nFate distribution:")
    for f, c in counts.items():
        print(f"  {f.value:12s}: {c:6d} ({c/n*100:5.1f}%)")
    
    return {
        'curvature': mean_curv,
        'delta_mech': delta_mech,
        'vol_strain': vol_strain,
        'fate': fate,
        'counts': counts,
        'exceeds_threshold': exceeds,
    }


def visualize_delta_distribution(delta_mech: np.ndarray,
                                  fate: np.ndarray,
                                  title: str = "δ_mech Distribution") -> plt.Figure:
    """
    δ分布を可視化（Tail Shift）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. δ分布（log scale）
    ax1 = axes[0]
    bins = np.linspace(0, 0.30, 60)
    
    ax1.hist(delta_mech, bins=bins, alpha=0.7, color='skyblue',
             edgecolor='black', linewidth=0.5, label='All vertices')
    
    # Tail部分をハイライト
    tail = delta_mech[delta_mech > 0.10]
    if len(tail) > 0:
        ax1.hist(tail, bins=bins, alpha=0.9, color='red',
                edgecolor='darkred', linewidth=0.8,
                label=f'δ > 0.10 ({len(tail)/len(delta_mech)*100:.2f}%)')
    
    ax1.axvline(0.10, color='black', linestyle='--', linewidth=2,
               label='Lindemann threshold')
    ax1.axvline(delta_mech.mean(), color='blue', linestyle=':',
               linewidth=2, label=f'Mean = {delta_mech.mean():.4f}')
    
    ax1.set_xlabel('δ_mech', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_xlim([0, 0.30])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('δ_mech Distribution\n(Mechanical Lindemann ratio from curvature)')
    
    # 2. Fate分布
    ax2 = axes[1]
    fate_names = ['STABLE', 'PLASTIC', 'CRACK']
    fate_counts = [(fate == AtomFate[f]).sum() for f in fate_names]
    colors = ['green', 'yellow', 'red']
    
    bars = ax2.bar(fate_names, fate_counts, color=colors, edgecolor='black')
    
    for bar, count in zip(bars, fate_counts):
        pct = count / len(fate) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Fate Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    from mesh_loader import load_dxf
    
    print("Loading Step 9 mesh...")
    mesh = load_dxf('/mnt/user-data/uploads/step9.dxf')
    
    # SPM別に解析
    results = {}
    for SPM in [10, 20, 30]:
        print(f"\n{'='*60}")
        print(f"SPM = {SPM}")
        print('='*60)
        
        result = analyze_cad_shape(
            mesh.vertices, mesh.faces, mesh.Z,
            thickness_mm=1.96,
            T=300.0,
            SPM=SPM
        )
        results[SPM] = result
    
    # 可視化
    fig = visualize_delta_distribution(
        results[20]['delta_mech'],
        results[20]['fate'],
        title='Step 9: δ_mech Analysis (SPM=20)'
    )
    fig.savefig('/mnt/user-data/outputs/step9_delta_mech.png', dpi=150)
    print(f"\nSaved: /mnt/user-data/outputs/step9_delta_mech.png")
    
    # 3D可視化
    fig2 = plt.figure(figsize=(16, 6))
    
    # δ_mech マップ
    ax1 = fig2.add_subplot(121, projection='3d')
    delta = results[20]['delta_mech']
    scatter = ax1.scatter(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                          c=delta, cmap='hot', s=1, vmin=0, vmax=0.15)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_zlabel('Z [mm]')
    ax1.set_title('δ_mech Distribution')
    plt.colorbar(scatter, ax=ax1, label='δ_mech', shrink=0.6)
    
    # Fate マップ
    ax2 = fig2.add_subplot(122, projection='3d')
    fate = results[20]['fate']
    colors = np.array(['green' if f == AtomFate.STABLE else
                       'yellow' if f == AtomFate.PLASTIC else
                       'red' for f in fate])
    ax2.scatter(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                c=colors, s=1, alpha=0.6)
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_zlabel('Z [mm]')
    ax2.set_title('Fate Map (Green=OK, Red=CRACK)')
    
    plt.suptitle('Step 9: Curvature → δ_mech → Fate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig('/mnt/user-data/outputs/step9_delta_3d.png', dpi=150)
    print(f"Saved: /mnt/user-data/outputs/step9_delta_3d.png")
    
    # サマリ
    print("\n" + "="*60)
    print("SUMMARY: SPM vs CRACK Risk")
    print("="*60)
    for SPM in [10, 20, 30]:
        n = len(mesh.vertices)
        crack = results[SPM]['counts'][AtomFate.CRACK]
        plastic = results[SPM]['counts'][AtomFate.PLASTIC]
        print(f"SPM={SPM}: CRACK {crack/n*100:.2f}%, PLASTIC {plastic/n*100:.2f}%")
