"""
Λ-Dynamics POC: NIDEC Motor Case Press Forming Analysis
========================================================

Nidec中国工場のモータケース（SECD t=1.96mm）の
プレス成形過程をΛ³/U²理論で解析するPOC

実行: python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

# ローカルモジュール
from materials import get_material
from physics_engine import PhysicsEngine, create_engine
from mesh_loader import load_dxf, PressMesh
from strain_calc import estimate_strain_from_geometry
from u2_engine import U2Engine, MultiStepAnalyzer


def visualize_risk_map(results: list, output_path: str):
    """
    リスクマップを可視化
    
    λ = U²/U²_c のカラーマップ
    緑: 安全 (λ < 0.5)
    黄: 注意 (0.5 ≤ λ < 0.8)
    赤: 危険 (0.8 ≤ λ < 1.0)
    黒: 破壊 (λ ≥ 1.0)
    """
    n_steps = len(results)
    
    # 3x3グリッド（9ステップ対応）
    n_cols = 3
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(18, 6 * n_rows))
    
    # カラーマップ（安全→危険）
    cmap = plt.cm.RdYlGn_r  # 緑→黄→赤
    
    for i, res in enumerate(results):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        
        mesh = res['mesh']
        risk = res['risk_map']
        
        # 頂点カラーを計算
        risk_clipped = np.clip(risk, 0, 1.5)
        
        # 面ごとの平均リスク
        face_risk = np.array([risk_clipped[face].mean() for face in mesh.faces])
        
        # カラーマッピング
        norm = mcolors.Normalize(vmin=0, vmax=1.2)
        colors = cmap(norm(face_risk))
        
        # 崩壊した面は黒
        collapsed_faces = np.array([res['engine'].collapsed[face].any() 
                                    for face in mesh.faces])
        colors[collapsed_faces] = [0, 0, 0, 1]  # 黒
        
        # 面を描画（サンプリング）
        faces_to_draw = []
        colors_to_draw = []
        
        n_faces = len(mesh.faces)
        if n_faces > 2000:
            indices = np.random.choice(n_faces, 2000, replace=False)
        else:
            indices = np.arange(n_faces)
        
        for idx in indices:
            face = mesh.faces[idx]
            verts = mesh.vertices[face]
            faces_to_draw.append(verts)
            colors_to_draw.append(colors[idx])
        
        poly = Poly3DCollection(faces_to_draw, alpha=0.8)
        poly.set_facecolor(colors_to_draw)
        poly.set_edgecolor('gray')
        poly.set_linewidth(0.1)
        ax.add_collection3d(poly)
        
        # 軸設定
        all_pts = mesh.vertices
        max_range = max(
            all_pts[:,0].max() - all_pts[:,0].min(),
            all_pts[:,1].max() - all_pts[:,1].min(),
            all_pts[:,2].max() - all_pts[:,2].min()
        ) / 2
        
        mid = all_pts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        
        summary = res['summary']
        # Step 9に★マークを付ける
        crack_mark = "★CRACK" if summary['step'] == 9 else ""
        title = (f"Step {summary['step']}{crack_mark}: "
                f"λ_max={summary['lambda_max']:.3f}\n"
                f"t_min={summary['thickness_min']:.2f}mm, "
                f"High risk: {summary['high_risk_ratio']*100:.1f}%")
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=20, azim=45)
    
    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('λ = U²/U²_c (Risk Factor)', fontsize=12)
    cbar.ax.axhline(y=1.0, color='black', linewidth=2)
    cbar.ax.text(1.5, 1.0, 'FAILURE', fontsize=8, va='center')
    
    plt.suptitle('Λ-Dynamics Risk Map: NIDEC Motor Case (SECD t=1.96mm) - 9 Steps', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")


def plot_lambda_history(results: list, output_path: str):
    """λの履歴をプロット"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    steps = [r['summary']['step'] for r in results]
    
    # λ統計
    ax = axes[0]
    lambda_max = [r['summary']['lambda_max'] for r in results]
    lambda_mean = [r['summary']['lambda_mean'] for r in results]
    
    ax.plot(steps, lambda_max, 'ro-', label='λ_max', linewidth=2, markersize=8)
    ax.plot(steps, lambda_mean, 'bo-', label='λ_mean', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Failure threshold')
    ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=1, label='High risk threshold')
    ax.set_xlabel('Process Step')
    ax.set_ylabel('λ = U²/U²_c')
    ax.set_title('Lambda Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(lambda_max) * 1.2)
    
    # 高リスク比率
    ax = axes[1]
    high_risk = [r['summary']['high_risk_ratio'] * 100 for r in results]
    ax.bar(steps, high_risk, color='orangered', alpha=0.7)
    ax.set_xlabel('Process Step')
    ax.set_ylabel('High Risk Area [%]')
    ax.set_title('High Risk (λ > 0.8) Area Ratio')
    ax.grid(True, alpha=0.3)
    
    # 歪み
    ax = axes[2]
    strain_max = [r['summary']['strain_max'] for r in results]
    strain_mean = [r['summary']['strain_mean'] for r in results]
    
    ax.plot(steps, np.array(strain_max)*100, 'g^-', label='ε_max', linewidth=2, markersize=8)
    ax.plot(steps, np.array(strain_mean)*100, 'gs-', label='ε_mean', linewidth=2, markersize=8)
    ax.set_xlabel('Process Step')
    ax.set_ylabel('Strain [%]')
    ax.set_title('Strain Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Λ-Dynamics Analysis: Process History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    """POCメイン実行"""
    print("="*70)
    print("Λ-Dynamics POC: NIDEC Motor Case Press Forming Analysis")
    print("="*70)
    
    # ファイルパス（9ステップ）
    files = [
        "/mnt/user-data/uploads/step1.dxf",
        "/mnt/user-data/uploads/step2.dxf",
        "/mnt/user-data/uploads/step3.dxf",
        "/mnt/user-data/uploads/step4.dxf",
        "/mnt/user-data/uploads/step5.dxf",
        "/mnt/user-data/uploads/step6.dxf",
        "/mnt/user-data/uploads/step7.dxf",
        "/mnt/user-data/uploads/step8.dxf",
        "/mnt/user-data/uploads/step9.dxf",  # ★亀裂発生工程！
    ]
    
    step_names = [
        "①ブランク",
        "②胴体1絞り",
        "③胴体2絞り", 
        "④胴体3絞り",
        "⑤逆絞り1",
        "⑥逆絞り2",
        "⑦逆絞り3",
        "⑧胴体4絞り",
        "⑨胴体5絞り★CRACK",  # ★亀裂発生！
    ]
    
    # 1. メッシュ読み込み
    print("\n[1] Loading DXF meshes...")
    meshes = []
    for i, f in enumerate(files):
        mesh = load_dxf(f)
        meshes.append(mesh)
        print(f"  Step {i+1} ({step_names[i]}): {mesh.n_vertices} vertices, {mesh.n_faces} faces")
    
    # 2. 歪み履歴を計算
    print("\n[2] Computing strain history...")
    strain_history = estimate_strain_from_geometry(meshes)
    
    for info in strain_history:
        print(f"  Step {info['step']}: ε_equiv_max = {info['eps_equiv_max']:.3f}")
    
    # 3. 材料設定
    print("\n[3] Setting up material (SECD t=1.96mm)...")
    physics = create_engine('SECD')
    mat = physics.mat
    thickness = 1.96  # mm
    
    print(f"  Material: {mat['name']} ({mat['structure']})")
    print(f"  Structure: {mat['structure']}, Z_bulk: {mat['Z_bulk']}")
    print(f"  δ_L: {mat['delta_L']}")
    print(f"  FLC₀ (t={thickness}mm): {physics.compute_FLC0(thickness)*100:.1f}%")
    print(f"  n-value: {mat['n_value']}, r-value: {mat['r_value']}")
    
    # 4. U²解析実行
    print("\n[4] Running Λ-Dynamics analysis...")
    analyzer = MultiStepAnalyzer(physics, thickness_mm=thickness)
    results = analyzer.analyze_process(meshes, strain_history)
    
    # 5. 結果サマリ
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY (with CUMULATIVE STRAIN)")
    print("="*70)
    print(f"\n{'Step':<6} {'λ_max':<8} {'ε_cum':<8} {'t_min':<8} {'thin%':<8} {'FAIL%':<8} {'Risk%':<8}")
    print("-"*62)
    for r in results:
        s = r['summary']
        cum_strain = s.get('cumulative_strain_max', s['strain_max'])
        fail_ratio = s.get('failure_ratio', 0) * 100
        print(f"{s['step']:<6} {s['lambda_max']:<8.3f} {cum_strain:<8.3f} "
              f"{s['thickness_min']:<8.2f} {s['thinning_max']:<8.1f} "
              f"{fail_ratio:<8.1f} {s['high_risk_ratio']*100:<8.1f}")
    
    # 6. 危険箇所の特定
    print("\n[5] Identifying critical regions...")
    final_result = results[-1]
    final_risk = final_result['risk_map']
    final_mesh = final_result['mesh']
    
    # 高リスク頂点の位置を特定
    high_risk_mask = final_risk > 0.8
    if high_risk_mask.any():
        high_risk_positions = final_mesh.vertices[high_risk_mask]
        print(f"\n  High risk vertices (λ > 0.8): {high_risk_mask.sum()}")
        print(f"  Location (centroid): "
              f"X={high_risk_positions[:,0].mean():.1f}, "
              f"Y={high_risk_positions[:,1].mean():.1f}, "
              f"Z={high_risk_positions[:,2].mean():.1f} mm")
        
        # Y方向（深さ）での分布
        y_range = (high_risk_positions[:,1].min(), high_risk_positions[:,1].max())
        print(f"  Y range (depth): {y_range[0]:.1f} to {y_range[1]:.1f} mm")
    
    # 7. 可視化
    print("\n[6] Generating visualizations...")
    visualize_risk_map(results, '/mnt/user-data/outputs/lambda_risk_map.png')
    plot_lambda_history(results, '/mnt/user-data/outputs/lambda_history.png')
    
    print("\n" + "="*70)
    print("POC COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print("  - /mnt/user-data/outputs/lambda_risk_map.png")
    print("  - /mnt/user-data/outputs/lambda_history.png")
    
    return results


if __name__ == "__main__":
    results = main()
