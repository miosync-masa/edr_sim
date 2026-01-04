"""
Λ³-Dynamics Tensile Test Simulator (v2)
========================================

シンプルな直方体金属ブロックの引張試験

統合物理エンジン使用:
  - Born collapse（熱軟化）
  - Debye-Waller（熱ゆらぎ）
  - Hooke（弾性変形）
  - Lindemann criterion（臨界判定）

Author: Tamaki & Masamichi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum

# 統合物理エンジンをインポート
from physics_core import PhysicsEngine, MaterialPhysics

# CuPyが使えればGPU、なければCPU
try:
    import cupy as cp
    USE_GPU = True
    print("✓ CuPy available - GPU mode")
except ImportError:
    import numpy as cp
    USE_GPU = False
    print("✗ CuPy not available - CPU mode")


class AtomFate(Enum):
    STABLE = "STABLE"
    PLASTIC = "PLASTIC"
    CRACK = "CRACK"
    NECKING = "NECKING"


class TensileTestSimulator:
    """
    引張試験シミュレータ（v2）
    
    統合物理エンジン使用:
    - Born collapse
    - Debye-Waller
    - Hooke
    - Lindemann
    """
    
    def __init__(self, 
                 Lx: float = 10.0,  # mm
                 Ly: float = 50.0,  # mm  
                 Lz: float = 10.0,  # mm
                 spacing: float = 1.0,  # mm (シミュレーション格子)
                 material: MaterialPhysics = None):
        """
        Args:
            Lx, Ly, Lz: 試験片サイズ [mm]
            spacing: 格子点間隔 [mm]
            material: 材料（MaterialPhysics）
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.spacing = spacing
        
        # 物理エンジン初期化
        self.mat = material or MaterialPhysics.FCC_Cu()  # 銅でテスト（検証済み）
        self.physics = PhysicsEngine(self.mat)
        
        print(f"\nSpecimen: {Lx} x {Ly} x {Lz} mm")
        print(f"Grid spacing: {spacing} mm")
        
        # 格子点を生成
        self._create_lattice()
        
        # 近傍リストを構築
        self._build_neighbors()
        
        # Z_effを計算
        self._compute_Z_eff()
        
        # 初期状態を保存
        self.X = self.positions.copy()  # 参照座標
        self.R0 = self._compute_all_vectors(self.X)  # 参照ベクトル
        
        # 累積量
        self.U2_cumul = np.zeros(self.N)
        self.strain_history = []
        self.stress_history = []
        
        print(f"Lattice points: {self.N}")
        print(f"Z_eff range: [{self.Z_eff.min():.1f}, {self.Z_eff.max():.1f}]")
    
    def _create_lattice(self):
        """格子点を生成"""
        nx = int(self.Lx / self.spacing) + 1
        ny = int(self.Ly / self.spacing) + 1
        nz = int(self.Lz / self.spacing) + 1
        
        x = np.linspace(0, self.Lx, nx)
        y = np.linspace(0, self.Ly, ny)
        z = np.linspace(0, self.Lz, nz)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        self.N = len(self.positions)
        
        # 境界条件用のインデックス
        tol = self.spacing * 0.1
        self.fixed_bottom = np.where(self.positions[:, 1] < tol)[0]
        self.loaded_top = np.where(self.positions[:, 1] > self.Ly - tol)[0]
        
        print(f"  Fixed nodes (bottom): {len(self.fixed_bottom)}")
        print(f"  Loaded nodes (top): {len(self.loaded_top)}")
    
    def _build_neighbors(self):
        """近傍リストを構築"""
        cutoff = self.spacing * 1.8  # 最近接 + α
        tree = cKDTree(self.positions)
        
        self.neighbors = []
        self.n_neighbors = np.zeros(self.N, dtype=int)
        
        for i in range(self.N):
            nb = tree.query_ball_point(self.positions[i], cutoff)
            nb = [j for j in nb if j != i]
            self.neighbors.append(np.array(nb))
            self.n_neighbors[i] = len(nb)
        
        self.k_max = max(self.n_neighbors)
        print(f"  Max neighbors: {self.k_max}")
    
    def _compute_Z_eff(self):
        """有効配位数を計算"""
        # 表面/エッジ/コーナー判定
        tol = self.spacing * 0.1
        
        self.Z_eff = np.full(self.N, self.mat.Z_bulk, dtype=float)
        
        for i in range(self.N):
            x, y, z = self.positions[i]
            
            # 境界にいくつ接しているか
            n_boundaries = 0
            if x < tol or x > self.Lx - tol: n_boundaries += 1
            if y < tol or y > self.Ly - tol: n_boundaries += 1
            if z < tol or z > self.Lz - tol: n_boundaries += 1
            
            if n_boundaries == 0:
                self.Z_eff[i] = self.mat.Z_bulk  # バルク
            elif n_boundaries == 1:
                self.Z_eff[i] = self.mat.Z_bulk * 0.75  # 面
            elif n_boundaries == 2:
                self.Z_eff[i] = self.mat.Z_bulk * 0.5   # エッジ
            else:
                self.Z_eff[i] = self.mat.Z_bulk * 0.375 # コーナー
    
    def _compute_all_vectors(self, positions: np.ndarray) -> np.ndarray:
        """全近傍ベクトルを計算"""
        R = np.zeros((self.N, self.k_max, 3))
        
        for i in range(self.N):
            for j_idx, j in enumerate(self.neighbors[i]):
                R[i, j_idx] = positions[j] - positions[i]
        
        return R
    
    def compute_deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        変形勾配テンソル F を計算
        
        F = (Σ r ⊗ R₀) · (Σ R₀ ⊗ R₀)^(-1)
        
        r: 現在の相対ベクトル
        R₀: 参照の相対ベクトル
        """
        R = self._compute_all_vectors(x)
        F = np.zeros((self.N, 3, 3))
        
        for i in range(self.N):
            k = self.n_neighbors[i]
            if k < 3:
                F[i] = np.eye(3)
                continue
            
            r = R[i, :k]   # [k, 3]
            r0 = self.R0[i, :k]  # [k, 3]
            
            # Σ r ⊗ R₀
            rR0 = r.T @ r0  # [3, 3]
            
            # Σ R₀ ⊗ R₀
            R0R0 = r0.T @ r0  # [3, 3]
            
            # F = rR₀ · R₀R₀^(-1)
            try:
                F[i] = rR0 @ np.linalg.inv(R0R0)
            except:
                F[i] = np.eye(3)
        
        return F
    
    def compute_strain_tensor(self, F: np.ndarray) -> np.ndarray:
        """
        Green-Lagrangeひずみテンソル E を計算
        
        E = (F^T F - I) / 2
        """
        I = np.eye(3)
        C = np.einsum('nij,nkj->nik', F, F)  # C = F^T F
        E = (C - I) / 2
        return E
    
    def compute_U2_mechanical(self, E: np.ndarray) -> np.ndarray:
        """
        機械的U²を計算
        
        U²_mech = ||E||² × a²
        """
        # ひずみテンソルのフロベニウスノルム²
        E_norm_sq = np.sum(E**2, axis=(1, 2))
        
        # 原子スケールに変換
        # メッシュのε → 原子の変位²
        a = self.material.a
        U2_mech = E_norm_sq * a**2
        
        return U2_mech
    
    def compute_U2_thermal(self, T: float) -> np.ndarray:
        """
        熱的U²を計算
        
        ⟨u²⟩_th ≈ 3 k_B T / (m ω²)
        
        簡易計算: δ² × a² × (T / T_melt)
        """
        a = self.material.a
        delta_L = self.material.delta_L
        T_melt = self.material.T_melt
        
        # 融点でδ = δ_L になるようスケーリング
        U2_th = (delta_L * a)**2 * (T / T_melt)
        
        return np.full(self.N, U2_th)
    
    def compute_lambda(self, U2_total: np.ndarray) -> np.ndarray:
        """
        λ = U² / U²_c を計算
        
        U²_c = (δ_L × a)² × (Z/Z_bulk)³
        """
        a = self.material.a
        delta_L = self.material.delta_L
        Z_bulk = self.material.Z_bulk
        
        U2_c_bulk = (delta_L * a)**2
        Z_ratio = self.Z_eff / Z_bulk
        U2_c = U2_c_bulk * (Z_ratio ** 3)
        
        return U2_total / np.maximum(U2_c, 1e-30)
    
    def compute_volumetric_strain(self, E: np.ndarray) -> np.ndarray:
        """体積ひずみ（三軸度）を計算"""
        return np.trace(E, axis1=1, axis2=2)
    
    def determine_fate(self, 
                        lam: np.ndarray, 
                        vol_strain: np.ndarray,
                        strain_rate: float = 0.001) -> np.ndarray:
        """
        運命を判定
        
        λ ≥ 1 かつ:
          引張 (vol > 0) → CRACK
          圧縮 (vol < 0) → PLASTIC
          中間 (|vol| ≈ 0) → NECKING (純粋せん断)
        """
        fate = np.full(self.N, AtomFate.STABLE, dtype=object)
        
        # 閾値を小さく: 引張試験でのポアソン効果を考慮
        # vol_strain ≈ ε(1 - 2ν) ≈ 0.3ε for ν≈0.35
        threshold = 0.001  # 0.1% 以上の体積変化でCRACK/PLASTIC判定
        
        unstable = lam >= 1.0
        
        for i in np.where(unstable)[0]:
            if vol_strain[i] > threshold:
                fate[i] = AtomFate.CRACK
            elif vol_strain[i] < -threshold:
                fate[i] = AtomFate.PLASTIC
            else:
                fate[i] = AtomFate.NECKING
        
        return fate
    
    def apply_displacement(self, displacement_y: float):
        """
        Y方向に変位を適用（引張）
        
        下端固定、上端を引っ張る
        """
        x = self.positions.copy()
        
        # Y座標に応じて変位を線形補間
        y_normalized = x[:, 1] / self.Ly
        x[:, 1] += displacement_y * y_normalized
        
        # ポアソン効果（横収縮）
        nu = 0.3
        x[:, 0] -= displacement_y * nu * (x[:, 0] - self.Lx/2) / self.Ly
        x[:, 2] -= displacement_y * nu * (x[:, 2] - self.Lz/2) / self.Ly
        
        self.positions = x
    
    def run_step(self, displacement_y: float, T: float = 300.0) -> dict:
        """
        1ステップを実行（物理エンジン統合版）
        
        Args:
            displacement_y: Y方向変位 [mm]
            T: 温度 [K]
        
        Returns:
            解析結果
        """
        # 1. 変位を適用
        self.apply_displacement(displacement_y)
        
        # 2. 変形勾配テンソル
        F = self.compute_deformation_gradient(self.positions)
        
        # 3. ひずみテンソル
        E = self.compute_strain_tensor(F)
        
        # 4. 物理エンジンで U² 計算
        U2_total, U2_thermal, U2_mech = self.physics.total_U2(E, T)
        
        # 注: 引張試験では累積しない（参照状態からの変形を直接見る）
        # プレス成形では累積が必要だが、引張は参照→現在の直接比較
        
        # 6. λ計算（物理エンジン）
        U2_c = self.physics.critical_U2(self.Z_eff)
        lam = U2_total / np.maximum(U2_c, 1e-30)
        
        # 7. 体積ひずみ
        vol_strain = self.compute_volumetric_strain(E)
        
        # 8. 運命判定
        engineering_strain = displacement_y / self.Ly
        fate = self.determine_fate(lam, vol_strain, engineering_strain)
        
        # 9. 統計
        counts = {f: (fate == f).sum() for f in AtomFate}
        
        # 10. 応力推定（温度依存のヤング率使用！）
        E_modulus = self.physics.youngs_modulus(T)
        mean_strain = np.mean(np.abs(E[:, 1, 1]))  # Y方向ひずみ
        stress = E_modulus * mean_strain
        
        # 履歴保存
        total_strain = (self.positions[self.loaded_top, 1].mean() - self.Ly) / self.Ly
        self.strain_history.append(total_strain)
        self.stress_history.append(stress)
        
        # δ（Lindemann比）
        delta_thermal = self.physics.thermal_lindemann_ratio(T)
        r_nn = self.physics.nearest_neighbor_distance(T)
        delta_mech = np.sqrt(U2_mech) / r_nn
        
        return {
            'F': F,
            'E': E,
            'U2_mech': U2_mech,
            'U2_thermal': U2_thermal,
            'U2_total': U2_total,
            'U2_c': U2_c,
            'lambda': lam,
            'vol_strain': vol_strain,
            'fate': fate,
            'counts': counts,
            'strain': total_strain,
            'stress': stress,
            'delta_thermal': delta_thermal,
            'delta_mech': delta_mech,
            'T': T,
        }
    
    def run_full_test(self, 
                       max_strain: float = 0.3,
                       n_steps: int = 30,
                       T: float = 300.0,
                       stop_on_fracture: bool = True) -> List[dict]:
        """
        引張試験をフル実行
        
        Args:
            stop_on_fracture: Falseならヒートマップ用に最後まで走らせる
        """
        print(f"\n{'='*60}")
        print(f"Running tensile test: max_strain={max_strain}, steps={n_steps}")
        print(f"Temperature: {T} K")
        print(f"{'='*60}")
        
        results = []
        disp_per_step = max_strain * self.Ly / n_steps
        
        # CAD連携用：各原子が初めてλ>1になったステップを記録
        first_unstable_step = np.full(self.N, -1, dtype=int)  # -1 = never unstable
        
        for step in range(n_steps):
            result = self.run_step(disp_per_step, T)
            results.append(result)
            
            # 進捗表示
            strain = result['strain'] * 100
            lam = result['lambda']
            lam_max = lam.max()
            lam_mean = lam.mean()
            
            # λ > 1 の割合
            frac_unstable = (lam > 1.0).sum() / len(lam) * 100
            n_crack = result['counts'][AtomFate.CRACK]
            
            # 初めてλ>1になった原子を記録（ヒートマップ用）
            newly_unstable = (lam >= 1.0) & (first_unstable_step < 0)
            first_unstable_step[newly_unstable] = step + 1  # 1-indexed
            
            if step % 5 == 0 or frac_unstable > 50:
                print(f"Step {step+1:3d}: ε={strain:5.1f}%, λ_max={lam_max:.2f}, "
                      f"λ_mean={lam_mean:.2f}, unstable={frac_unstable:.0f}%")
            
            # 破断判定: 50%以上の原子が λ > 1
            if frac_unstable > 50:
                print(f"\n*** FRACTURE at ε = {strain:.1f}% (>50% atoms unstable) ***")
                if stop_on_fracture:
                    break
        
        # 最終結果にヒートマップ情報を追加
        if results:
            results[-1]['first_unstable_step'] = first_unstable_step
            results[-1]['n_steps_run'] = len(results)
            results[-1]['total_steps'] = n_steps
        
        return results
    
    def plot_results(self, results: List[dict]):
        """結果を可視化"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 応力-ひずみ曲線
        ax1 = fig.add_subplot(231)
        strains = [r['strain']*100 for r in results]
        stresses = [r['stress']/1e6 for r in results]
        ax1.plot(strains, stresses, 'b-o', linewidth=2)
        ax1.set_xlabel('Strain [%]')
        ax1.set_ylabel('Stress [MPa]')
        ax1.set_title('Stress-Strain Curve')
        ax1.grid(True, alpha=0.3)
        
        # 降伏応力をマーク（MaterialPhysicsには sigma_y がないので省略）
        # ax1.axhline(y=self.mat.sigma_y/1e6, color='r', linestyle='--',
        #            label=f'σ_y = {self.mat.sigma_y/1e6:.0f} MPa')
        ax1.legend()
        
        # 2. λ_max vs ひずみ
        ax2 = fig.add_subplot(232)
        lam_maxs = [r['lambda'].max() for r in results]
        ax2.plot(strains, lam_maxs, 'r-o', linewidth=2)
        ax2.axhline(y=1.0, color='k', linestyle='--', label='λ=1 (Critical)')
        ax2.set_xlabel('Strain [%]')
        ax2.set_ylabel('λ_max')
        ax2.set_title('Maximum λ vs Strain')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. CRACK数 vs ひずみ
        ax3 = fig.add_subplot(233)
        cracks = [r['counts'][AtomFate.CRACK] for r in results]
        ax3.plot(strains, cracks, 'r-o', linewidth=2)
        ax3.set_xlabel('Strain [%]')
        ax3.set_ylabel('CRACK count')
        ax3.set_title('Crack Propagation')
        ax3.grid(True, alpha=0.3)
        
        # 4. 最終状態の3D表示（λ）
        ax4 = fig.add_subplot(234, projection='3d')
        final = results[-1]
        lam = final['lambda']
        scatter = ax4.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2],
                              c=lam, cmap='hot', s=20, vmin=0, vmax=2)
        ax4.set_xlabel('X [mm]')
        ax4.set_ylabel('Y [mm]')
        ax4.set_zlabel('Z [mm]')
        ax4.set_title(f'λ Distribution (ε={strains[-1]:.1f}%)')
        plt.colorbar(scatter, ax=ax4, label='λ', shrink=0.6)
        
        # 5. 最終状態の3D表示（Fate）
        ax5 = fig.add_subplot(235, projection='3d')
        fate = final['fate']
        colors = np.array(['green' if f == AtomFate.STABLE else
                          'yellow' if f == AtomFate.PLASTIC else
                          'orange' if f == AtomFate.NECKING else
                          'red' for f in fate])
        ax5.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2],
                    c=colors, s=20, alpha=0.7)
        ax5.set_xlabel('X [mm]')
        ax5.set_ylabel('Y [mm]')
        ax5.set_zlabel('Z [mm]')
        ax5.set_title('Fate Map\n(Green=OK, Yellow=Plastic, Red=Crack)')
        
        # 6. 断面のλ分布
        ax6 = fig.add_subplot(236)
        # Y-Z断面（X中央）
        x_mid = self.Lx / 2
        tol = self.spacing * 0.6
        mid_slice = np.abs(self.positions[:, 0] - x_mid) < tol
        
        if mid_slice.sum() > 0:
            y_slice = self.positions[mid_slice, 1]
            lam_slice = final['lambda'][mid_slice]
            ax6.scatter(y_slice, lam_slice, c='blue', alpha=0.5)
            ax6.axhline(y=1.0, color='r', linestyle='--', label='λ=1')
            ax6.set_xlabel('Y position [mm]')
            ax6.set_ylabel('λ')
            ax6.set_title('λ along specimen length')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Λ³-Dynamics Tensile Test: {self.mat.name}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Λ³-DYNAMICS TENSILE TEST SIMULATION (v2)")
    print("With Born Collapse + Debye-Waller + Hooke")
    print("="*70)
    
    # 銅の引張試験（物理エンジン検証済み）
    sim = TensileTestSimulator(
        Lx=10.0,   # mm
        Ly=50.0,   # mm
        Lz=10.0,   # mm
        spacing=2.0,  # mm
        material=MaterialPhysics.FCC_Cu()
    )
    
    # 引張試験実行（室温）
    print("\n### Room Temperature Test (T=300K) ###")
    results_300K = sim.run_full_test(
        max_strain=0.50,  # 50%まで
        n_steps=50,
        T=300.0
    )
    
    # 結果を可視化
    fig = sim.plot_results(results_300K)
    fig.savefig('/mnt/user-data/outputs/tensile_test_v2_300K.png', dpi=150)
    print(f"\nSaved: /mnt/user-data/outputs/tensile_test_v2_300K.png")
    
    # 高温試験
    print("\n### High Temperature Test (T=1000K) ###")
    sim2 = TensileTestSimulator(
        Lx=10.0, Ly=50.0, Lz=10.0,
        spacing=2.0,
        material=MaterialPhysics.FCC_Cu()
    )
    
    results_1000K = sim2.run_full_test(
        max_strain=0.50,
        n_steps=50,
        T=1000.0  # 高温！
    )
    
    fig2 = sim2.plot_results(results_1000K)
    fig2.savefig('/mnt/user-data/outputs/tensile_test_v2_1000K.png', dpi=150)
    print(f"Saved: /mnt/user-data/outputs/tensile_test_v2_1000K.png")
    
    # サマリ
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n300K:")
    final_300 = results_300K[-1]
    print(f"  Final strain: {final_300['strain']*100:.1f}%")
    print(f"  λ_max: {final_300['lambda'].max():.3f}")
    print(f"  δ_thermal: {final_300['delta_thermal']:.4f}")
    print(f"  CRACK count: {final_300['counts'][AtomFate.CRACK]}")
    
    print("\n1000K:")
    final_1000 = results_1000K[-1]
    print(f"  Final strain: {final_1000['strain']*100:.1f}%")
    print(f"  λ_max: {final_1000['lambda'].max():.3f}")
    print(f"  δ_thermal: {final_1000['delta_thermal']:.4f}")
    print(f"  CRACK count: {final_1000['counts'][AtomFate.CRACK]}")
    
    print(f"\n→ 高温(1000K)ではδ_thermalが大きい分、早く破断するはず！")
