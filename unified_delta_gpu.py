#!/usr/bin/env python3
"""
Unified δ-Theory GPU Engine (CuPy + Sparse)
=============================================

100³〜200³格子対応のGPU高速版

スパース化のポイント：
  - 近傍リスト → CSR行列
  - カスケード伝播 → SpMV (Sparse Matrix-Vector)
  - δ計算 → ベクトル化

Author: Tamaki & Masamichi
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum

# CuPy（なければNumPyにフォールバック）
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    from cupyx.scipy.sparse import csr_matrix as cp_csr
    GPU_AVAILABLE = True
    print("✓ CuPy available - GPU mode")
except ImportError:
    import numpy as cp
    from scipy.sparse import csr_matrix as cp_csr
    import scipy.sparse as cpsparse
    GPU_AVAILABLE = False
    print("✗ CuPy not available - CPU fallback")

# 物理定数
k_B = 1.380649e-23
u_kg = 1.66053906660e-27


class DeformationPhase(Enum):
    HOOKE = 0
    NONLINEAR = 1
    YIELD = 2
    PLASTIC = 3
    FAILURE = 4


@dataclass
class MaterialGPU:
    """GPU用材料データ（軽量版）"""
    name: str
    structure: str
    Z_bulk: int
    a_300K: float
    alpha: float
    E0: float
    nu: float
    T_melt: float
    M_amu: float
    delta_L: float
    lambda_base: float
    kappa: float
    E_bond_eV: float
    
    @classmethod
    def SECD(cls):
        return cls(
            name="SECD", structure="BCC", Z_bulk=8,
            a_300K=2.87e-10, alpha=1.5e-5,
            E0=210e9, nu=0.29, T_melt=1811,
            M_amu=55.845, delta_L=0.18,
            lambda_base=49.2, kappa=0.573,
            E_bond_eV=4.28,
        )
    
    @classmethod
    def FCC_Cu(cls):
        return cls(
            name="FCC-Cu", structure="FCC", Z_bulk=12,
            a_300K=3.61e-10, alpha=1.7e-5,
            E0=130e9, nu=0.34, T_melt=1357,
            M_amu=63.546, delta_L=0.10,
            lambda_base=26.3, kappa=1.713,
            E_bond_eV=3.49,
        )


class SparseNeighborGraph:
    """
    スパース近傍グラフ（CSR形式）
    
    CSR (Compressed Sparse Row):
      indptr:  各行の開始位置
      indices: 列インデックス（近傍ID）
      data:    重み（距離など、今回は1.0）
    
    利点:
      - メモリ効率: O(N × k_avg) vs O(N × k_max)
      - SpMV高速: カスケード伝播が O(edges)
    """
    
    def __init__(self, N: int, neighbors_list: list):
        """
        Args:
            N: 頂点数
            neighbors_list: 各頂点の近傍リスト
        """
        self.N = N
        
        # CSR構築
        indptr = [0]
        indices = []
        data = []
        
        for i in range(N):
            nb = neighbors_list[i] if i < len(neighbors_list) else []
            for j in nb:
                indices.append(j)
                data.append(1.0)
            indptr.append(len(indices))
        
        # NumPy配列に
        self.indptr = np.array(indptr, dtype=np.int32)
        self.indices = np.array(indices, dtype=np.int32)
        self.data = np.array(data, dtype=np.float32)
        
        # GPU転送
        if GPU_AVAILABLE:
            self.indptr_gpu = cp.asarray(self.indptr)
            self.indices_gpu = cp.asarray(self.indices)
            self.data_gpu = cp.asarray(self.data)
            self.csr_gpu = cpsparse.csr_matrix(
                (self.data_gpu, self.indices_gpu, self.indptr_gpu),
                shape=(N, N)
            )
        else:
            from scipy.sparse import csr_matrix
            self.csr_gpu = csr_matrix(
                (self.data, self.indices, self.indptr),
                shape=(N, N)
            )
        
        # 統計
        self.n_edges = len(indices)
        self.k_avg = self.n_edges / N if N > 0 else 0
        
        print(f"SparseNeighborGraph: N={N}, edges={self.n_edges}, k_avg={self.k_avg:.1f}")
    
    def get_neighbors(self, i: int) -> np.ndarray:
        """頂点iの近傍を取得（CPU）"""
        start = self.indptr[i]
        end = self.indptr[i + 1]
        return self.indices[start:end]
    
    def propagate(self, values: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        値を近傍に伝播（SpMV）
        
        Args:
            values: 各頂点の値 [N]
            mask: 伝播元マスク [N] (Noneなら全部)
        
        Returns:
            propagated: 各頂点が受け取った値の合計 [N]
        """
        if mask is not None:
            values = values * mask.astype(values.dtype)
        
        if GPU_AVAILABLE:
            values_gpu = cp.asarray(values)
            result = self.csr_gpu.T @ values_gpu  # 転置してSpMV
            return cp.asnumpy(result)
        else:
            return self.csr_gpu.T @ values


class UnifiedDeltaGPU:
    """
    GPU対応統一δエンジン
    
    全計算をベクトル化してGPU並列実行
    
    熱揺らぎ確率：
      σ_δ = δ_thermal（揺らぎの幅）
      P_exceed = exp(-(δ_L - δ) / σ_δ)
      
      δ < δ_L でも確率的に崩壊可能！
      これがクリープ/疲労の物理
    """
    
    DELTA_HOOKE = 0.01
    DELTA_NONLINEAR = 0.03
    DELTA_YIELD = 0.05
    
    # Z依存融点のスケーリング指数
    ALPHA_MELT = 1.2
    
    # 熱揺らぎ定数
    NU_0 = 1e13  # Debye周波数（試行頻度）[Hz]
    
    def __init__(self, material: MaterialGPU):
        self.mat = material
        self.M = material.M_amu * u_kg
        
        self.G0 = material.E0 / (2.0 * (1.0 + material.nu))
        self.K0 = material.E0 / (3.0 * (1.0 - 2.0 * material.nu))
        
        # 結合エネルギー
        self.E_bond = material.E_bond_eV * 1.602176e-19
        
        print(f"UnifiedDeltaGPU: {material.name}")
        print(f"  GPU: {GPU_AVAILABLE}")
    
    # ========================================
    # ベクトル化された計算（GPU対応）
    # ========================================
    
    def thermal_softening_vec(self, T: np.ndarray) -> np.ndarray:
        """熱軟化（ベクトル版）"""
        xp = cp if GPU_AVAILABLE else np
        T = xp.asarray(T)
        
        T_ref = 293.0
        delta_T = xp.maximum(T - T_ref, 0)
        lambda_eff = self.mat.lambda_base * (1.0 + self.mat.kappa * delta_T / 1000.0)
        
        result = xp.exp(-lambda_eff * self.mat.alpha * delta_T)
        
        return cp.asnumpy(result) if GPU_AVAILABLE else result
    
    def youngs_modulus_vec(self, T: np.ndarray) -> np.ndarray:
        """ヤング率（ベクトル版）"""
        soft = self.thermal_softening_vec(T)
        return self.mat.E0 * soft
    
    def delta_thermal_vec(self, T: np.ndarray) -> np.ndarray:
        """δ_thermal（ベクトル版）"""
        xp = cp if GPU_AVAILABLE else np
        T = xp.asarray(T)
        
        # 簡易計算: δ_th ≈ 0.017 × √(T/300) × (E0/E(T))^0.5
        # これはDebye-Wallerの近似
        T_ref = 300.0
        soft = xp.asarray(self.thermal_softening_vec(cp.asnumpy(T) if GPU_AVAILABLE else T))
        
        # 基準値（300Kでのδ_thermal）
        delta_300K = 0.017 * (self.mat.E0 / 130e9) ** 0.3  # Cu基準でスケール
        
        result = delta_300K * xp.sqrt(T / T_ref) / xp.sqrt(xp.maximum(soft, 0.01))
        
        return cp.asnumpy(result) if GPU_AVAILABLE else result
    
    def delta_mechanical_vec(self, sigma_local: np.ndarray, T: np.ndarray) -> np.ndarray:
        """δ_mech（ベクトル版）"""
        E_T = self.youngs_modulus_vec(T)
        return np.abs(sigma_local) / np.maximum(E_T, 1e6)
    
    def delta_total_vec(self, sigma_local: np.ndarray, T: np.ndarray) -> np.ndarray:
        """合計δ（ベクトル版）"""
        return self.delta_thermal_vec(T) + self.delta_mechanical_vec(sigma_local, T)
    
    # ========================================
    # 熱揺らぎ確率（クリープ/疲労の物理）
    # Arrhenius則 + 活性化エネルギー
    # ========================================
    
    def fluctuation_sigma_vec(self, T: np.ndarray) -> np.ndarray:
        """
        熱揺らぎの幅 σ_δ（ベクトル版）
        
        σ_δ ≈ δ_thermal
        
        高温ほど揺らぎが大きい
        """
        return self.delta_thermal_vec(T)
    
    def activation_energy_vec(self, delta: np.ndarray, Z_eff: np.ndarray = None) -> np.ndarray:
        """
        活性化エネルギー E_a（ベクトル版）
        
        δ → δ_L に行くための「山」の高さ
        
        E_a = E_bond × Z_eff × (1 - δ/δ_L)²
        
        物理的意味:
          - δ ≈ 0 → E_a ≈ E_bond × Z（全結合を切る）
          - δ → δ_L → E_a → 0（臨界状態、障壁なし）
          - Z低い → E_a低い（切る結合が少ない）
        
        Args:
            delta: 現在のδ [N]
            Z_eff: 有効配位数 [N]（Noneならバルク）
        
        Returns:
            E_a [N]: 活性化エネルギー [J]
        """
        if Z_eff is None:
            Z_eff = np.full_like(delta, self.mat.Z_bulk)
        
        # 正規化: δ/δ_L（0〜1で臨界）
        delta_ratio = np.clip(delta / self.mat.delta_L, 0, 1)
        
        # 障壁高さ: (1 - δ/δ_L)²
        barrier_factor = (1.0 - delta_ratio) ** 2
        
        # E_a = E_bond × (Z_eff / Z_bulk) × barrier
        E_a = self.E_bond * (Z_eff / self.mat.Z_bulk) * barrier_factor
        
        return E_a
    
    def jump_rate_vec(self,
                       delta: np.ndarray,
                       T: np.ndarray,
                       Z_eff: np.ndarray = None) -> np.ndarray:
        """
        熱活性化ジャンプレート（Arrhenius則）
        
        rate = ν₀ × exp(-E_a / kT)
        
        Args:
            delta: 現在のδ [N]
            T: 温度 [N]
            Z_eff: 有効配位数 [N]
        
        Returns:
            rate [N]: ジャンプレート [1/s]
        """
        E_a = self.activation_energy_vec(delta, Z_eff)
        kT = k_B * np.maximum(T, 1.0)
        
        # Arrhenius
        exponent = -E_a / kT
        exponent = np.clip(exponent, -100, 0)  # オーバーフロー防止
        
        return self.NU_0 * np.exp(exponent)
    
    def probability_exceed_vec(self, 
                                delta: np.ndarray, 
                                T: np.ndarray,
                                Z_eff: np.ndarray = None) -> np.ndarray:
        """
        熱揺らぎで閾値を超える「瞬間」確率
        
        Boltzmann分布の裾野:
        P = exp(-E_a / kT)
        
        これは「1試行で超える確率」
        実際の崩壊レートは rate = ν₀ × P
        
        Returns:
            P_exceed [N]: 確率 [0, 1]
        """
        E_a = self.activation_energy_vec(delta, Z_eff)
        kT = k_B * np.maximum(T, 1.0)
        
        # 既に臨界以上なら確率1
        at_critical = delta >= self.mat.delta_L
        
        exponent = -E_a / kT
        exponent = np.clip(exponent, -100, 0)
        
        P = np.where(at_critical, 1.0, np.exp(exponent))
        
        return np.clip(P, 0, 1)
    
    def stochastic_collapse_mask(self,
                                  delta: np.ndarray,
                                  T: np.ndarray,
                                  Z_eff: np.ndarray = None,
                                  dt: float = 1e-6) -> np.ndarray:
        """
        確率的崩壊マスク（モンテカルロ）
        
        時間dtの間に熱揺らぎで崩壊するサイトを決定
        
        rate = ν₀ × exp(-E_a / kT)
        P_collapse_in_dt = 1 - exp(-rate × dt)
        
        Args:
            delta: 現在のδ [N]
            T: 温度 [N]
            Z_eff: 有効配位数 [N]
            dt: 時間ステップ [s]
        
        Returns:
            collapse_mask [N]: bool
        """
        rate = self.jump_rate_vec(delta, T, Z_eff)
        
        # dt間の崩壊確率（Poisson過程）
        P_collapse = 1.0 - np.exp(-rate * dt)
        
        # モンテカルロサンプリング
        random = np.random.random(len(delta))
        
        # 決定論的崩壊（δ ≥ δ_L）も含める
        deterministic = delta >= self.mat.delta_L
        
        return deterministic | (random < P_collapse)
    
    def expected_lifetime_vec(self,
                               delta: np.ndarray,
                               T: np.ndarray,
                               Z_eff: np.ndarray = None) -> np.ndarray:
        """
        期待寿命（クリープ/疲労寿命）
        
        τ = 1 / rate = (1/ν₀) × exp(E_a / kT)
        
        δ → δ_L に近いほど E_a 小 → 寿命短い
        高温ほど kT 大 → 寿命短い
        Z小 → E_a 小 → 寿命短い
        
        Returns:
            lifetime [N]: 秒
        """
        rate = self.jump_rate_vec(delta, T, Z_eff)
        rate = np.maximum(rate, 1e-30)  # ゼロ除算防止
        
        return 1.0 / rate
    
    # ========================================
    # 融点・相判定
    # ========================================
    
    def local_melting_temperature_vec(self, Z_eff: np.ndarray) -> np.ndarray:
        """Z依存融点（ベクトル版）"""
        Z_ratio = np.clip(Z_eff / self.mat.Z_bulk, 0.1, 1.0)
        return self.mat.T_melt * (Z_ratio ** self.ALPHA_MELT)
    
    def is_molten_vec(self, T: np.ndarray, Z_eff: np.ndarray) -> np.ndarray:
        """融解判定（ベクトル版）"""
        T_melt_local = self.local_melting_temperature_vec(Z_eff)
        return T > T_melt_local
    
    def determine_phase_vec(self, delta: np.ndarray) -> np.ndarray:
        """相判定（ベクトル版）→ 整数で返す"""
        phase = np.zeros(len(delta), dtype=np.int32)
        phase[delta >= self.DELTA_HOOKE] = 1     # NONLINEAR
        phase[delta >= self.DELTA_NONLINEAR] = 2  # YIELD  
        phase[delta >= self.DELTA_YIELD] = 3      # PLASTIC
        phase[delta >= self.mat.delta_L] = 4      # FAILURE
        return phase


class CascadeGPU:
    """
    GPU対応カスケードエンジン
    
    スパース行列でカスケード伝播を高速化
    """
    
    def __init__(self, 
                 material: MaterialGPU, 
                 graph: SparseNeighborGraph,
                 efficiency: float = 0.1):
        self.mat = material
        self.graph = graph
        self.engine = UnifiedDeltaGPU(material)
        self.efficiency = efficiency
        
        # 1結合あたりの発熱
        E_bond = material.E_bond_eV * 1.602176e-19
        self.dT_per_bond = E_bond / (3 * k_B) * efficiency
        
        print(f"CascadeGPU: ΔT/bond = {self.dT_per_bond:.1f} K")
    
    def cascade_step_gpu(self,
                         delta: np.ndarray,
                         T: np.ndarray,
                         Z: np.ndarray,
                         sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        GPUカスケードステップ
        
        SpMVで発熱を近傍に伝播
        """
        N = len(delta)
        
        # 崩壊マスク
        collapsed = delta >= self.mat.delta_L
        n_collapsed = np.sum(collapsed)
        
        if n_collapsed == 0:
            return delta, T, Z, 0
        
        # 発熱を近傍に伝播（SpMV！）
        heat_source = collapsed.astype(np.float32) * self.dT_per_bond
        heat_received = self.graph.propagate(heat_source)
        
        # 温度更新
        T_new = T + heat_received
        T_new = np.clip(T_new, 0, self.mat.T_melt * 10)  # 上限
        
        # Z低下（崩壊した近傍の数だけ）
        Z_loss = self.graph.propagate(collapsed.astype(np.float32))
        Z_new = np.maximum(Z - Z_loss, 0.5)
        
        # δ再計算
        K_t = self.mat.Z_bulk / np.maximum(Z_new, 0.5)
        sigma_local = sigma * K_t
        delta_new = self.engine.delta_total_vec(sigma_local, T_new)
        
        # 新しく崩壊した数
        collapsed_new = delta_new >= self.mat.delta_L
        n_new = np.sum(collapsed_new) - n_collapsed
        
        return delta_new, T_new, Z_new, max(n_new, 0)
    
    def run_cascade(self,
                    delta: np.ndarray,
                    T: np.ndarray,
                    Z: np.ndarray,
                    sigma: np.ndarray,
                    max_iterations: int = 50) -> Dict:
        """
        カスケードを収束まで実行（GPU高速版）
        """
        delta = delta.copy()
        T = T.copy()
        Z = Z.copy()
        
        history = [np.sum(delta >= self.mat.delta_L)]
        
        for it in range(max_iterations):
            delta, T, Z, n_new = self.cascade_step_gpu(delta, T, Z, sigma)
            
            history.append(np.sum(delta >= self.mat.delta_L))
            
            if n_new == 0:
                break
        
        # 融解判定
        molten = self.engine.is_molten_vec(T, Z)
        
        return {
            'delta': delta,
            'T': T,
            'Z': Z,
            'history': history,
            'iterations': it + 1,
            'collapsed': np.sum(delta >= self.mat.delta_L),
            'molten': molten,
            'white_layer_frac': np.mean(molten),
            'T_max': T.max(),
        }


def build_3d_lattice_graph(Nx: int, Ny: int, Nz: int) -> Tuple[np.ndarray, SparseNeighborGraph]:
    """
    3D格子の近傍グラフを構築
    
    Args:
        Nx, Ny, Nz: 格子サイズ
    
    Returns:
        positions: [N, 3]
        graph: SparseNeighborGraph
    """
    N = Nx * Ny * Nz
    print(f"Building 3D lattice: {Nx}×{Ny}×{Nz} = {N} sites")
    
    # 位置
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    
    # 近傍リスト（6近傍: ±x, ±y, ±z）
    def idx(i, j, k):
        if 0 <= i < Nx and 0 <= j < Ny and 0 <= k < Nz:
            return i * Ny * Nz + j * Nz + k
        return -1
    
    neighbors = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                nb = []
                for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                    n = idx(i+di, j+dj, k+dk)
                    if n >= 0:
                        nb.append(n)
                neighbors.append(nb)
    
    graph = SparseNeighborGraph(N, neighbors)
    
    return positions, graph


# ========================================
# テスト
# ========================================
if __name__ == "__main__":
    print("="*60)
    print("Unified δ-Theory GPU Engine Test")
    print("="*60)
    
    # 小さい格子でテスト
    Nx, Ny, Nz = 20, 20, 20
    positions, graph = build_3d_lattice_graph(Nx, Ny, Nz)
    N = len(positions)
    
    # 材料
    mat = MaterialGPU.SECD()
    engine = UnifiedDeltaGPU(mat)
    cascade = CascadeGPU(mat, graph, efficiency=0.1)
    
    # 初期条件
    T_init = np.full(N, 500.0, dtype=np.float32)
    Z_init = np.full(N, 8.0, dtype=np.float32)
    sigma_init = np.full(N, 1000e6, dtype=np.float32)  # 1 GPa
    
    # 表面のZ低下
    surface_mask = (
        (positions[:, 0] == 0) | (positions[:, 0] == Nx-1) |
        (positions[:, 1] == 0) | (positions[:, 1] == Ny-1) |
        (positions[:, 2] == 0) | (positions[:, 2] == Nz-1)
    )
    Z_init[surface_mask] = 4.0
    
    # コーナーはさらに低Z
    corner_mask = (
        ((positions[:, 0] == 0) | (positions[:, 0] == Nx-1)).astype(int) +
        ((positions[:, 1] == 0) | (positions[:, 1] == Ny-1)).astype(int) +
        ((positions[:, 2] == 0) | (positions[:, 2] == Nz-1)).astype(int)
    ) >= 2
    Z_init[corner_mask] = 3.0
    
    print(f"\nInitial state:")
    print(f"  N = {N}")
    print(f"  T = 500 K")
    print(f"  σ = 1000 MPa")
    print(f"  Z: bulk={np.sum(Z_init==8)}, surface={np.sum(Z_init==4)}, corner={np.sum(Z_init==3)}")
    
    # δ計算
    K_t = mat.Z_bulk / np.maximum(Z_init, 0.5)
    sigma_local = sigma_init * K_t
    delta_init = engine.delta_total_vec(sigma_local, T_init)
    
    print(f"\n  δ_thermal = {engine.delta_thermal_vec(T_init)[0]:.4f}")
    print(f"  δ_total range: [{delta_init.min():.4f}, {delta_init.max():.4f}]")
    print(f"  Initially collapsed: {np.sum(delta_init >= mat.delta_L)}")
    
    # シード注入（中央に1点）
    center = N // 2
    delta_seed = delta_init.copy()
    delta_seed[center] = 0.20
    
    print(f"\n--- Cascade from single seed ---")
    
    import time
    t0 = time.time()
    result = cascade.run_cascade(delta_seed, T_init.copy(), Z_init.copy(), sigma_init)
    elapsed = time.time() - t0
    
    print(f"  Time: {elapsed*1000:.1f} ms")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Collapsed: {result['collapsed']} / {N}")
    print(f"  T_max: {result['T_max']:.1f} K")
    print(f"  White layer: {result['white_layer_frac']*100:.1f}%")
    
    if result['collapsed'] > 1:
        print(f"\n  🔥 CASCADE!")
    
    # 大規模テスト
    print("\n" + "="*60)
    print("Large Scale Test")
    print("="*60)
    
    for size in [50, 100]:
        print(f"\n--- {size}³ lattice ---")
        t0 = time.time()
        pos, grp = build_3d_lattice_graph(size, size, size)
        t_build = time.time() - t0
        print(f"  Build time: {t_build:.2f} s")
        
        N = len(pos)
        cascade_big = CascadeGPU(mat, grp, efficiency=0.1)
        
        T = np.full(N, 500.0, dtype=np.float32)
        Z = np.full(N, 6.0, dtype=np.float32)  # 平均的にやや低Z
        sigma = np.full(N, 800e6, dtype=np.float32)
        
        K_t = mat.Z_bulk / Z
        sigma_loc = sigma * K_t
        delta = engine.delta_total_vec(sigma_loc, T)
        
        # シード
        delta[N//2] = 0.20
        
        t0 = time.time()
        res = cascade_big.run_cascade(delta, T, Z, sigma, max_iterations=10)
        t_cascade = time.time() - t0
        
        print(f"  Cascade time: {t_cascade:.2f} s ({res['iterations']} iterations)")
        print(f"  Collapsed: {res['collapsed']} ({res['collapsed']/N*100:.2f}%)")
        print(f"  Performance: {N / t_cascade / 1e6:.2f} M sites/s")
