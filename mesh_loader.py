"""
Λ-Dynamics Mesh Loader
======================
DXFファイルからメッシュを読み込み、頂点と面を抽出
"""

import numpy as np
import ezdxf
from typing import Tuple, List, Dict
from scipy.spatial import cKDTree


class PressMesh:
    """プレス成形メッシュクラス"""
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Args:
            vertices: 頂点座標 (N, 3) [mm]
            faces: 面インデックス (M, 3) - 三角形
        """
        self.vertices = vertices  # [mm]
        self.faces = faces
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        
        # 配位数を計算
        self.Z = self._compute_coordination()
        
        # 表面フラグ
        self.is_surface = self._detect_surface()
        
    def _compute_coordination(self, cutoff_factor: float = 1.5) -> np.ndarray:
        """
        各頂点の配位数（接続頂点数）を計算
        
        メッシュの接続性から配位数を推定
        """
        # 各頂点に接続する頂点を集める
        connections = [set() for _ in range(self.n_vertices)]
        
        for face in self.faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        connections[face[i]].add(face[j])
        
        Z = np.array([len(conn) for conn in connections])
        return Z
    
    def _detect_surface(self) -> np.ndarray:
        """
        表面頂点を検出
        
        配位数が少ない頂点を表面とみなす
        """
        Z_median = np.median(self.Z)
        return self.Z < Z_median
    
    def get_bounding_box(self) -> Dict:
        """バウンディングボックスを取得"""
        return {
            'min': self.vertices.min(axis=0),
            'max': self.vertices.max(axis=0),
            'size': self.vertices.max(axis=0) - self.vertices.min(axis=0),
            'center': (self.vertices.max(axis=0) + self.vertices.min(axis=0)) / 2,
        }
    
    def compute_face_normals(self) -> np.ndarray:
        """各面の法線ベクトルを計算"""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-10)
        
        return normals
    
    def compute_vertex_normals(self) -> np.ndarray:
        """各頂点の法線ベクトルを計算（面法線の平均）"""
        face_normals = self.compute_face_normals()
        
        vertex_normals = np.zeros((self.n_vertices, 3))
        
        for i, face in enumerate(self.faces):
            for v_idx in face:
                vertex_normals[v_idx] += face_normals[i]
        
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / np.maximum(norms, 1e-10)
        
        return vertex_normals
    
    def __repr__(self):
        bbox = self.get_bounding_box()
        return (f"PressMesh(vertices={self.n_vertices}, faces={self.n_faces}, "
                f"size={bbox['size']} mm)")


def load_dxf(filepath: str) -> PressMesh:
    """
    DXFファイルからメッシュを読み込む
    
    Args:
        filepath: DXFファイルパス
    
    Returns:
        PressMesh: メッシュオブジェクト
    """
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    
    # 3DFACEを収集
    raw_faces = []
    for entity in msp:
        if entity.dxftype() == '3DFACE':
            v0 = np.array(entity.dxf.vtx0)
            v1 = np.array(entity.dxf.vtx1)
            v2 = np.array(entity.dxf.vtx2)
            v3 = np.array(entity.dxf.vtx3)
            
            # 三角形に分割
            if np.allclose(v2, v3):
                raw_faces.append([v0, v1, v2])
            else:
                raw_faces.append([v0, v1, v2])
                raw_faces.append([v0, v2, v3])
    
    if not raw_faces:
        raise ValueError(f"No 3DFACE entities found in {filepath}")
    
    # 頂点をユニーク化
    all_vertices = np.array([v for face in raw_faces for v in face])
    
    # KD-Treeで重複頂点をマージ
    unique_vertices, vertex_map = _merge_vertices(all_vertices, tolerance=1e-6)
    
    # 面インデックスを再構築
    faces = []
    idx = 0
    for raw_face in raw_faces:
        face_indices = []
        for _ in range(3):
            face_indices.append(vertex_map[idx])
            idx += 1
        faces.append(face_indices)
    
    faces = np.array(faces)
    
    return PressMesh(unique_vertices, faces)


def _merge_vertices(vertices: np.ndarray, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    重複頂点をマージ
    
    Args:
        vertices: 全頂点 (N, 3)
        tolerance: 同一頂点とみなす距離
    
    Returns:
        unique_vertices: ユニーク頂点 (M, 3)
        vertex_map: 元インデックス → 新インデックスのマッピング
    """
    tree = cKDTree(vertices)
    
    n_vertices = len(vertices)
    vertex_map = np.arange(n_vertices)
    processed = np.zeros(n_vertices, dtype=bool)
    
    unique_list = []
    new_index = 0
    
    for i in range(n_vertices):
        if processed[i]:
            continue
        
        # 近傍頂点を探索
        neighbors = tree.query_ball_point(vertices[i], tolerance)
        
        # 同じグループにマージ
        for j in neighbors:
            vertex_map[j] = new_index
            processed[j] = True
        
        unique_list.append(vertices[i])
        new_index += 1
    
    return np.array(unique_list), vertex_map


def load_process_steps(filepaths: List[str]) -> List[PressMesh]:
    """
    複数工程のDXFを読み込む
    
    Args:
        filepaths: DXFファイルパスのリスト
    
    Returns:
        List[PressMesh]: 各工程のメッシュ
    """
    meshes = []
    for i, fp in enumerate(filepaths):
        print(f"Loading step {i+1}: {fp}")
        mesh = load_dxf(fp)
        meshes.append(mesh)
        print(f"  → {mesh}")
    
    return meshes


if __name__ == "__main__":
    # テスト
    import sys
    
    test_files = [
        "/mnt/user-data/uploads/step1.dxf",
        "/mnt/user-data/uploads/step2.dxf",
    ]
    
    for fp in test_files:
        print(f"\n{'='*60}")
        mesh = load_dxf(fp)
        print(mesh)
        print(f"  Z (coordination): min={mesh.Z.min()}, max={mesh.Z.max()}, mean={mesh.Z.mean():.1f}")
        print(f"  Surface vertices: {mesh.is_surface.sum()} / {mesh.n_vertices}")
        bbox = mesh.get_bounding_box()
        print(f"  Bounding box: {bbox['size']} mm")
