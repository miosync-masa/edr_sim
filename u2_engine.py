"""
Λ-Dynamics U² Engine
=====================
U²ベースの材料安定性判定エンジン

核心:
  λ = K / |V| ≈ U² / U²_c
  λ ≥ 1 で不安定（破壊、相転移）
"""

import numpy as np
from typing import Optional, Tuple
from materials import get_material, PHYSICAL_CONSTANTS
from physics_engine import PhysicsEngine
from mesh_loader import PressMesh


class U2Engine:
    """
    U²計算エンジン
    
    連立方程式なし、完全並列計算可能
    
    FLCベースのスケーリング：
    - ε = FLC限界 のとき λ = 1（破断）
    - 板厚変化を追跡（薄くなるとFLC限界が下がる）
    """
    
    def __init__(self, engine: PhysicsEngine, mesh: PressMesh, thickness_mm: float = 1.96):
        """
        Args:
            engine: PhysicsEngineオブジェクト
            mesh: メッシュオブジェクト
            thickness_mm: 初期板厚 [mm]
        """
        self.engine = engine
        self.mat = engine.mat  # 材料データ（dict）
        self.mesh = mesh
        self.n_vertices = mesh.n_vertices
        self.thickness_initial = thickness_mm
        
        # 板厚分布（各頂点で異なる）
        self.thickness = np.ones(self.n_vertices) * thickness_mm
        
        # FLC₀を計算（初期板厚）
        self.flc0 = engine.compute_FLC0(thickness_mm)
        print(f"  FLC₀ (t={thickness_mm}mm): {self.flc0*100:.1f}%")
        
        # 状態変数（頂点ごと）
        self.T = np.ones(self.n_vertices) * 300.0  # 温度 [K]
        self.strain = np.zeros(self.n_vertices)    # メジャーひずみ ε₁
        self.strain_minor = np.zeros(self.n_vertices)  # マイナーひずみ ε₂
        self.strain_ratio = np.zeros(self.n_vertices)  # ひずみ比 β = ε₂/ε₁
        self.Z = mesh.Z.astype(float)              # 配位数
        self.collapsed = np.zeros(self.n_vertices, dtype=bool)  # 崩壊フラグ
        
        # 履歴（非Markov性のため重要！）
        self.U2_history = []
        self.lambda_history = []
        self.thickness_history = []
    
    def update_thickness(self):
        """
        板厚を更新（体積一定則）
        
        ε₁ + ε₂ + ε₃ = 0
        t = t₀ × exp(ε₃) = t₀ × exp(-(ε₁ + ε₂))
        """
        # 板厚ひずみ
        eps_thickness = -(self.strain + self.strain_minor)
        
        # 板厚更新
        self.thickness = self.thickness_initial * np.exp(eps_thickness)
        
        # 最小板厚を制限（物理的にゼロにはならない）
        self.thickness = np.maximum(self.thickness, 0.1)  # 最低0.1mm
        
        # 板厚減少率を記録
        thinning = (self.thickness_initial - self.thickness) / self.thickness_initial * 100
        
        return {
            'min_thickness': self.thickness.min(),
            'max_thinning': thinning.max(),
            'mean_thickness': self.thickness.mean(),
        }
        
    def compute_U2_thermal(self) -> np.ndarray:
        """
        熱振動による U²
        
        U²_thermal = k_B × T / (m × ω²)
        """
        k_B = PHYSICAL_CONSTANTS['k_B']
        m = self.mat['mass']
        omega = self.engine.mat['E0'] / (2 * (1 + self.engine.mat['nu'])) / self.engine.mat['rho']
        
        return k_B * self.T / (m * omega**2)
    
    def compute_U2_mechanical(self) -> np.ndarray:
        """
        機械的変形による U²
        
        U²_mech = ε² × a²
        
        ひずみが大きいほど原子変位が大きい
        """
        a = self.mat['a']  # 格子定数 [m]
        
        # ひずみ → 変位
        # δ/a ≈ ε (微小ひずみ近似)
        # U² = δ² = (ε × a)²
        
        return (self.strain * a)**2
    
    def compute_U2_total(self) -> np.ndarray:
        """総U²を計算"""
        U2_th = self.compute_U2_thermal()
        U2_mech = self.compute_U2_mechanical()
        
        return U2_th + U2_mech
    
    def compute_U2_critical(self) -> np.ndarray:
        """
        臨界U²を計算（Z³スケーリング）
        
        U²_c = (δ_L × a)² × fG / (Z/Z_bulk)³
        
        表面（Z小）ほどU²_c低い → 壊れやすい
        """
        return self.engine.critical_U2(self.Z)
    
    def compute_lambda(self) -> np.ndarray:
        """
        λ = U² / U²_c を計算（FLCベース + 板厚追跡）
        
        FLCを基準にスケーリング：
        - ε = FLC限界 のとき λ_strain = 1
        - 板厚が薄くなるとFLC限界が下がる → λが上がりやすい
        - 表面効果は補正因子として適用（Z³は強すぎる）
        
        λ < 1: 安定
        λ ≥ 1: 不安定（崩壊）
        """
        # 1. 板厚を更新
        self.update_thickness()
        
        # 2. ひずみ由来のλ（FLCベース、板厚考慮）
        # FLC限界を計算（各頂点の板厚を使用）
        flc_limit = self.engine.compute_FLC(self.thickness, self.strain_minor)
        
        # λ_strain = (|ε| / ε_limit)²
        lambda_strain = (np.abs(self.strain) / np.maximum(flc_limit, 0.01)) ** 2
        
        # 3. 表面効果（Z低→壊れやすい）- 穏やかに
        Z_bulk = self.mat['Z_bulk']
        Z_ratio = np.clip(self.Z / Z_bulk, 0.3, 1.5)  # 極端な値を抑制
        # Z_factor: 表面で最大1.3倍、バルクで1.0
        Z_factor = 1.0 + 0.3 * (1.0 - Z_ratio)
        Z_factor = np.clip(Z_factor, 1.0, 1.3)
        
        # 4. 熱効果（高温で弱くなる）- 小さい補正
        T_ref = 300.0  # 室温
        T_factor = 1.0 + 0.1 * (self.T - T_ref) / T_ref  # 10%/100K程度
        T_factor = np.clip(T_factor, 0.9, 1.5)
        
        # 5. 総合λ
        lambda_total = lambda_strain * Z_factor * T_factor
        
        return lambda_total
    
    def check_instability(self) -> np.ndarray:
        """
        不安定判定
        
        Returns:
            bool配列: Trueの箇所が不安定
        """
        lam = self.compute_lambda()
        # λ > 1.0 で崩壊（FLC超過）
        return (lam > 1.0) & ~self.collapsed
    
    def run_born_cascade(self, max_iterations: int = 100) -> int:
        """
        Born崩壊カスケードを実行
        
        1. λ ≥ 1 の箇所を検出
        2. 崩壊 → 周囲のZを減少
        3. Zが減ると U²_c が下がる → さらに崩壊
        4. 収束するまで繰り返し
        
        Returns:
            総崩壊頂点数
        """
        total_collapsed = 0
        
        for iteration in range(max_iterations):
            # 不安定箇所を検出
            unstable = self.check_instability()
            n_unstable = unstable.sum()
            
            if n_unstable == 0:
                break
            
            total_collapsed += n_unstable
            
            # 崩壊マーク
            self.collapsed[unstable] = True
            
            # 周囲のZを減少（メッシュ接続から）
            self._reduce_neighbor_Z(unstable)
            
            # 発熱（結合エネルギー解放）
            self._apply_heat_generation(unstable)
        
        return total_collapsed
    
    def _reduce_neighbor_Z(self, collapsed_mask: np.ndarray):
        """崩壊箇所の近傍のZを減少"""
        # メッシュの接続性から近傍を取得
        for face in self.mesh.faces:
            # 面内に崩壊頂点があれば、他の頂点のZを減らす
            face_collapsed = collapsed_mask[face].any()
            if face_collapsed:
                for v_idx in face:
                    if not self.collapsed[v_idx]:
                        self.Z[v_idx] = max(1, self.Z[v_idx] - 1)
    
    def _apply_heat_generation(self, collapsed_mask: np.ndarray):
        """
        崩壊による発熱
        
        結合エネルギーが熱として解放 → 局所温度上昇
        これがシアバンド/白層形成のメカニズム！
        """
        k_B = PHYSICAL_CONSTANTS['k_B']
        eV_to_J = PHYSICAL_CONSTANTS['eV_to_J']
        bond_energy = self.mat['bond_energy_eV'] * eV_to_J
        
        # ΔT = E_bond / (3 k_B)
        dT = bond_energy / (3 * k_B)
        
        # 崩壊箇所の温度上昇
        self.T[collapsed_mask] += dT * 0.1  # 効率係数
    
    def set_strain_field(self, strain: np.ndarray, strain_minor: np.ndarray = None,
                          strain_ratio: float = None):
        """ひずみ場を設定
        
        Args:
            strain: メジャーひずみ ε₁
            strain_minor: マイナーひずみ ε₂（Noneならstrain_ratioから計算）
            strain_ratio: ひずみ比 β = ε₂/ε₁（strain_minorがNoneの場合に使用）
        """
        self.strain = np.asarray(strain)
        
        if strain_minor is not None:
            self.strain_minor = np.asarray(strain_minor)
            # ひずみ比を計算（ゼロ除算回避）
            with np.errstate(divide='ignore', invalid='ignore'):
                self.strain_ratio = np.where(
                    np.abs(self.strain) > 1e-6,
                    self.strain_minor / self.strain,
                    -0.3  # デフォルト値（深絞り）
                )
        elif strain_ratio is not None:
            # strain_ratioからminor strainを計算
            self.strain_ratio = np.ones_like(self.strain) * strain_ratio
            self.strain_minor = self.strain * strain_ratio
        else:
            # 深絞りでは負のひずみ比（径方向圧縮）
            # 深絞り領域: β ≈ -0.3
            self.strain_ratio = np.ones_like(self.strain) * (-0.3)
            self.strain_minor = self.strain * (-0.3)
    
    def set_temperature_field(self, T: np.ndarray):
        """温度場を設定"""
        self.T = np.asarray(T)
    
    def get_risk_map(self) -> np.ndarray:
        """
        破壊リスクマップを取得
        
        risk = λ = U²/U²_c
        risk < 1: 安全（緑）
        risk ≈ 1: 危険（黄〜赤）
        risk > 1: 破壊（黒）
        """
        lam = self.compute_lambda()
        lam[self.collapsed] = np.inf  # 既に崩壊した箇所
        return lam
    
    def summary(self) -> dict:
        """状態サマリを取得"""
        lam = self.compute_lambda()
        U2_total = self.compute_U2_total()
        U2_c = self.compute_U2_critical()
        
        # 板厚減少率
        thinning = (self.thickness_initial - self.thickness) / self.thickness_initial * 100
        
        return {
            'n_vertices': self.n_vertices,
            'n_collapsed': self.collapsed.sum(),
            'lambda_mean': lam[~self.collapsed].mean() if (~self.collapsed).any() else 0,
            'lambda_max': lam[~self.collapsed].max() if (~self.collapsed).any() else 0,
            'lambda_min': lam[~self.collapsed].min() if (~self.collapsed).any() else 0,
            'T_mean': self.T.mean(),
            'T_max': self.T.max(),
            'strain_mean': np.abs(self.strain).mean(),
            'strain_max': np.abs(self.strain).max(),
            'U2_total_mean': U2_total.mean(),
            'U2_c_mean': U2_c.mean(),
            'Z_mean': self.Z.mean(),
            # 板厚情報
            'thickness_min': self.thickness.min(),
            'thickness_mean': self.thickness.mean(),
            'thinning_max': thinning.max(),  # 最大板厚減少率 [%]
        }


class MultiStepAnalyzer:
    """
    複数工程の解析器
    
    履歴依存性を考慮したU²累積を追跡
    **同一材料点の歪み累積**を実装
    """
    
    def __init__(self, material: Material, thickness_mm: float = 1.96):
        """
        Args:
            material: 材料オブジェクト
            thickness_mm: 初期板厚 [mm]
        """
        self.material = material
        self.thickness = thickness_mm
        self.step_results = []
    
    def _map_strain_to_next_mesh(self, prev_mesh: 'PressMesh', prev_strain: np.ndarray,
                                   prev_strain_minor: np.ndarray, prev_thickness: np.ndarray,
                                   next_mesh: 'PressMesh') -> tuple:
        """
        前工程の歪みを次工程のメッシュにマッピング
        
        最近傍探索で「同じ材料点」を対応させる
        
        Args:
            prev_mesh: 前工程のメッシュ
            prev_strain: 前工程のメジャー歪み
            prev_strain_minor: 前工程のマイナー歪み
            prev_thickness: 前工程の板厚
            next_mesh: 次工程のメッシュ
        
        Returns:
            (mapped_strain, mapped_strain_minor, mapped_thickness)
        """
        from scipy.spatial import cKDTree
        
        # 前工程の頂点でKD木を構築
        tree = cKDTree(prev_mesh.vertices)
        
        # 次工程の各頂点に最近傍をマッピング
        distances, indices = tree.query(next_mesh.vertices, k=1)
        
        # 歪みと板厚を引き継ぎ（距離が遠すぎる場合は0）
        max_distance = 50.0  # mm（許容マッピング距離）
        
        mapped_strain = np.where(distances < max_distance, 
                                  prev_strain[indices], 0.0)
        mapped_strain_minor = np.where(distances < max_distance,
                                        prev_strain_minor[indices], 0.0)
        mapped_thickness = np.where(distances < max_distance,
                                     prev_thickness[indices], self.thickness)
        
        return mapped_strain, mapped_strain_minor, mapped_thickness
    
    def analyze_process(self, meshes: list, strain_history: list) -> list:
        """
        全工程を解析（累積歪み追跡付き）
        
        Args:
            meshes: 工程順のメッシュリスト
            strain_history: 工程順の歪み情報リスト
        
        Returns:
            各工程の解析結果
        """
        results = []
        
        # 累積歪み（材料点追跡）
        cumulative_strain = None
        cumulative_strain_minor = None
        cumulative_thickness = None
        prev_mesh = None
        
        for i, (mesh, strain_info) in enumerate(zip(meshes, strain_history)):
            print(f"\n--- Analyzing Step {i+1} ---")
            
            # U²エンジンを初期化
            engine = U2Engine(self.material, mesh, self.thickness)
            
            # 前工程からの累積歪みをマッピング
            if prev_mesh is not None and cumulative_strain is not None:
                mapped_strain, mapped_strain_minor, mapped_thickness = \
                    self._map_strain_to_next_mesh(
                        prev_mesh, cumulative_strain, cumulative_strain_minor,
                        cumulative_thickness, mesh
                    )
                
                # 累積歪み = 前工程からの歪み + 今工程の増分歪み
                total_strain = mapped_strain + strain_info['vertex_strain']
                total_strain_minor = mapped_strain_minor + strain_info['vertex_strain'] * (-0.3)
                
                # 板厚も引き継ぎ（さらに薄くなる）
                engine.thickness = mapped_thickness
                
                print(f"  Cumulative strain: max={total_strain.max():.3f}, mean={total_strain.mean():.3f}")
            else:
                total_strain = strain_info['vertex_strain']
                total_strain_minor = strain_info['vertex_strain'] * (-0.3)
            
            # 歪み場を設定
            engine.set_strain_field(total_strain, strain_minor=total_strain_minor)
            
            # λ（リスク）を計算
            risk = engine.get_risk_map()
            
            # Born崩壊カスケード
            n_collapsed = engine.run_born_cascade()
            
            # サマリを取得
            summary = engine.summary()
            summary['step'] = i + 1
            summary['n_born_collapsed'] = n_collapsed
            
            # 累積歪み情報を追加
            summary['cumulative_strain_max'] = total_strain.max()
            summary['cumulative_strain_mean'] = total_strain.mean()
            
            # 高リスク領域を特定
            high_risk_mask = (risk > 0.8) & ~engine.collapsed
            summary['n_high_risk'] = high_risk_mask.sum()
            summary['high_risk_ratio'] = high_risk_mask.sum() / mesh.n_vertices
            
            # λ > 1（破壊）の領域
            failure_mask = risk > 1.0
            summary['n_failure'] = failure_mask.sum()
            summary['failure_ratio'] = failure_mask.sum() / mesh.n_vertices
            
            # 結果を保存
            results.append({
                'summary': summary,
                'risk_map': risk.copy(),
                'engine': engine,
                'mesh': mesh,
                'cumulative_strain': total_strain.copy(),
            })
            
            # 次工程のために保存
            cumulative_strain = total_strain.copy()
            cumulative_strain_minor = total_strain_minor.copy()
            cumulative_thickness = engine.thickness.copy()
            prev_mesh = mesh
            
            # 結果を表示
            print(f"  λ_mean: {summary['lambda_mean']:.4f}")
            print(f"  λ_max: {summary['lambda_max']:.4f}")
            print(f"  High risk (λ>0.8): {summary['high_risk_ratio']*100:.1f}%")
            print(f"  FAILURE (λ>1.0): {summary['failure_ratio']*100:.1f}%")
            print(f"  Born collapsed: {n_collapsed}")
        
        self.step_results = results
        return results


if __name__ == "__main__":
    from mesh_loader import load_dxf
    from strain_calc import estimate_strain_from_geometry
    
    # テスト
    files = [
        "/mnt/user-data/uploads/step1.dxf",
        "/mnt/user-data/uploads/step2.dxf",
        "/mnt/user-data/uploads/step3.dxf",
        "/mnt/user-data/uploads/step4.dxf",
        "/mnt/user-data/uploads/step5.dxf",
    ]
    
    # メッシュ読み込み
    print("Loading meshes...")
    meshes = [load_dxf(f) for f in files]
    
    # 歪み履歴を計算
    print("\nComputing strain history...")
    strain_history = estimate_strain_from_geometry(meshes)
    
    # 材料を設定
    material = Material('SECD')
    print(f"\nMaterial: {material}")
    
    # 解析実行
    analyzer = MultiStepAnalyzer(material)
    results = analyzer.analyze_process(meshes, strain_history)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
