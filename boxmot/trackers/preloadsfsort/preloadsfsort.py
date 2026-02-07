# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license
# PreloadSFSORT: SFSORT-based tracker with pre-registered person matching
# SFSORTをベースに、事前登録された人物のみを追跡するトラッカー

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import logger as LOGGER
from boxmot.utils.matching import linear_assignment


class TrackState:
    """トラックの状態を表す列挙型."""
    Active = 0
    Lost_Central = 1
    Lost_Marginal = 2


@dataclass(eq=False)
class Track:
    """PreloadSFSORT用の軽量トラックコンテナ."""

    bbox: np.ndarray
    last_frame: int
    track_id: int
    conf: float
    cls: int
    det_ind: int
    state: int = TrackState.Active
    registered_id: int = None  # 事前登録ID
    curr_feat: np.ndarray = None  # 現在のReID特徴量
    history_observations: list = field(default_factory=list)

    def __post_init__(self):
        """履歴リストの初期化."""
        if not self.history_observations:
            self.history_observations = [self.bbox.copy()]

    @property
    def id(self) -> int:
        """track_idのエイリアス."""
        return self.track_id

    @property
    def xyxy(self) -> np.ndarray:
        """bboxのエイリアス."""
        return self.bbox

    def update(self, box: np.ndarray, frame_id: int, conf: float, cls: int, det_ind: int) -> None:
        """マッチしたトラックを最新の検出で更新."""
        self.bbox = box
        self.state = TrackState.Active
        self.last_frame = frame_id
        self.conf = float(conf)
        self.cls = int(cls)
        self.det_ind = int(det_ind)
        self.history_observations.append(box.copy())


class PreloadSFSORT(BaseTracker):
    """
    PreloadSFSORT: SFSORTベースの事前登録人物追跡トラッカー.

    SFSORTの高速なアルゴリズムをベースに、事前登録された人物のみを追跡します。
    Kalmanフィルタを使用せず、BBSI（Bounding Box Similarity Index）コスト関数を使用して
    高速なマッチングを実現します。

    Parameters:
    - reid_weights (Path): ReIDモデルの重みファイルへのパス
    - device (torch.device): モデルを実行するデバイス
    - half (bool): 半精度（fp16）を使用するかどうか
    - registered_images_path (Path): 事前登録画像を含むフォルダへのパス
    - match_threshold (float): 事前登録画像とのマッチング閾値（0.0-1.0）

    SFSORT固有パラメータ:
    - high_th (float): 高信頼度検出の閾値
    - match_th_first (float): 第1段階マッチングの閾値
    - new_track_th (float): 新規トラック作成の閾値
    - low_th (float): 低信頼度検出の閾値
    - match_th_second (float): 第2段階マッチングの閾値
    - marginal_timeout (int): 周辺部ロストトラックのタイムアウト
    - central_timeout (int): 中央部ロストトラックのタイムアウト
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        # PreloadSFSORT固有パラメータ
        registered_images_path: Path = None,
        match_threshold: float = 0.5,
        # ReID最適化パラメータ
        reverify_interval: int = 30,  # 再確認間隔（フレーム数）
        overlap_iou_threshold: float = 0.3,  # 交差検出のIoU閾値
        # SFSORT固有パラメータ
        high_th: float = 0.6,
        match_th_first: float = 0.67,
        new_track_th: float = 0.7,
        low_th: float = 0.1,
        match_th_second: float = 0.3,
        dynamic_tuning: bool = False,
        cth: float = 0.5,
        high_th_m: float = 0.0,
        new_track_th_m: float = 0.0,
        match_th_first_m: float = 0.0,
        marginal_timeout: int = 30,
        central_timeout: int = 30,
        frame_width: int = None,
        frame_height: int = None,
        horizontal_margin: int = None,
        vertical_margin: int = None,
        **kwargs,
    ) -> None:
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        det_thresh = 0.6 if high_th is None else float(high_th)
        super().__init__(det_thresh=det_thresh, _tracker_name="PreloadSFSORT", **init_args, **kwargs)

        # SFSORTパラメータ
        self.high_th = self._resolve_or_default(high_th, 0.6, 0.0, 1.0)
        self.match_th_first = self._resolve_or_default(match_th_first, 0.67, 0.0, 0.67)
        self.new_track_th = self._resolve_or_default(new_track_th, 0.7, self.high_th, 1.0)
        self.low_th = self._resolve_or_default(low_th, 0.1, 0.0, self.high_th)
        self.match_th_second = self._resolve_or_default(match_th_second, 0.3, 0.0, 1.0)

        # 動的閾値調整
        self.dynamic_tuning = bool(dynamic_tuning)
        self.cth = self._resolve_or_default(cth, 0.5, self.low_th, 1.0)
        if self.dynamic_tuning:
            self.high_th_m = self._resolve_or_default(high_th_m, 0.0, 0.02, 0.1)
            self.new_track_th_m = self._resolve_or_default(new_track_th_m, 0.0, 0.02, 0.08)
            self.match_th_first_m = self._resolve_or_default(match_th_first_m, 0.0, 0.02, 0.08)
        else:
            self.high_th_m = 0.0 if high_th_m is None else float(high_th_m)
            self.new_track_th_m = 0.0 if new_track_th_m is None else float(new_track_th_m)
            self.match_th_first_m = 0.0 if match_th_first_m is None else float(match_th_first_m)

        # タイムアウト設定
        self.marginal_timeout = int(self._resolve_or_default(marginal_timeout, 30, 0, 500))
        self.central_timeout = int(self._resolve_or_default(central_timeout, 30, 0, 1000))

        # マージン設定
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.horizontal_margin = horizontal_margin
        self.vertical_margin = vertical_margin
        self.l_margin = 0.0
        self.r_margin = 0.0
        self.t_margin = 0.0
        self.b_margin = 0.0
        self._margins_ready = False
        self._maybe_set_margins(frame_width, frame_height)

        # トラック管理
        self.id_counter = 0
        self.active_tracks: list[Track] = []
        self.lost_stracks: list[Track] = []

        # ReID最適化パラメータ
        self.reverify_interval = reverify_interval
        self.overlap_iou_threshold = overlap_iou_threshold
        self.track_last_verified: dict[int, int] = {}  # {track_id: last_verified_frame}
        self.recovered_tracks: set[int] = set()  # ロストから復帰したトラックID

        # PreloadSFSORT固有: ReIDモデルと事前登録画像
        self.match_threshold = match_threshold
        self.registered_ids = {}  # {id: list of embeddings}
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        if registered_images_path:
            self._load_registered_images(Path(registered_images_path))

    def _load_registered_images(self, path: Path) -> None:
        """事前登録画像を読み込み、特徴量を抽出する（Winner-take-all方式）."""
        person_dirs = sorted(path.iterdir())

        for idx, person_dir in enumerate(person_dirs, start=1):
            if not person_dir.is_dir():
                continue

            # フォルダ名がIDとして使用可能な場合はそれを使用
            try:
                person_id = int(person_dir.name)
            except ValueError:
                person_id = idx

            embeddings = []
            for img_file in person_dir.glob('*.png'):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                bbox = np.array([[0, 0, w, h]])
                emb = self.model.get_features(bbox, img)
                emb_normalized = emb[0] / np.linalg.norm(emb[0])
                embeddings.append(emb_normalized)

            # JPGファイルも処理
            for img_file in person_dir.glob('*.jpg'):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                bbox = np.array([[0, 0, w, h]])
                emb = self.model.get_features(bbox, img)
                emb_normalized = emb[0] / np.linalg.norm(emb[0])
                embeddings.append(emb_normalized)

            if len(embeddings) == 0:
                continue

            self.registered_ids[person_id] = embeddings
            LOGGER.info(f"Registered person ID {person_id} with {len(embeddings)} images")

        LOGGER.info(f"Total registered persons: {len(self.registered_ids)}")

    def _match_detection_with_registered(self, feat: np.ndarray) -> tuple[int | None, float]:
        """検出を事前登録画像とマッチング（Winner-take-all方式）."""
        if feat is None:
            return None, 0.0

        best_match_id = None
        best_similarity = -1.0

        for reg_id, reg_embs in self.registered_ids.items():
            for reg_emb in reg_embs:
                similarity = np.dot(feat, reg_emb)
                if similarity > best_similarity and similarity >= self.match_threshold:
                    best_similarity = similarity
                    best_match_id = reg_id

        return best_match_id, best_similarity

    def _find_track_by_registered_id(self, registered_id: int) -> Track | None:
        """指定された登録IDを持つトラックを検索."""
        all_tracks = self.active_tracks + self.lost_stracks
        for track in all_tracks:
            if track.registered_id == registered_id:
                return track
        return None

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
        """フレームを処理し、トラック情報を更新（ReID最適化版）."""
        self.check_inputs(dets=dets, img=img, embs=embs)
        if self.is_obb:
            raise AssertionError("PreloadSFSORT does not support OBB detections")

        if not self._margins_ready and hasattr(self, "w") and hasattr(self, "h"):
            self._maybe_set_margins(self.w, self.h)

        self.frame_count += 1

        boxes = dets[:, :4] if dets.size else np.empty((0, 4))
        scores = dets[:, 4] if dets.size else np.empty((0,))
        classes = dets[:, 5] if dets.size else np.empty((0,))
        det_inds = np.arange(len(dets)) if dets.size else np.empty((0,), dtype=int)

        if len(boxes) == 0:
            # 検出がない場合
            self._purge_stale_lost_tracks()
            for track in self.active_tracks:
                self.lost_stracks.append(track)
                self._update_track_state_for_loss(track)
            self.active_tracks = []
            return np.empty((0, 8), dtype=float)

        # 動的閾値の計算
        hth, nth, mth = self._dynamic_thresholds(scores)

        next_active_tracks: list[Track] = []
        self._purge_stale_lost_tracks()

        track_pool = self.active_tracks + self.lost_stracks

        # ========================================
        # ステップ1: IoUベースのマッチング（ReIDなし）
        # ========================================
        matched_track_indices = set()
        matched_det_indices = set()
        recovered_track_ids = set()  # ロストから復帰したトラック

        if track_pool and len(boxes) > 0:
            cost = self.calculate_cost(track_pool, boxes)
            matches, unmatched_tracks_idx, unmatched_det_idx = linear_assignment(cost, mth)

            for track_idx, det_idx in matches:
                track = track_pool[track_idx]
                was_lost = track in self.lost_stracks

                track.update(
                    boxes[det_idx],
                    self.frame_count,
                    scores[det_idx],
                    classes[det_idx],
                    det_inds[det_idx],
                )
                next_active_tracks.append(track)
                matched_track_indices.add(track_idx)
                matched_det_indices.add(det_idx)

                if was_lost:
                    self.lost_stracks.remove(track)
                    recovered_track_ids.add(track.track_id)

            unmatched_det_indices = set(unmatched_det_idx)
        else:
            unmatched_det_indices = set(range(len(boxes)))

        # ========================================
        # ステップ2: 交差検出（IoUベース）
        # ========================================
        overlapping_track_ids = set()
        if len(next_active_tracks) > 1:
            overlapping_track_ids = self._detect_overlapping_tracks(next_active_tracks)

        # ========================================
        # ステップ3: ReIDが必要なトラック/検出を特定
        # ========================================
        # ReIDが必要な条件:
        # (a) 新規検出（IoU未マッチ）
        # (b) ロストから復帰したトラック
        # (c) 一定間隔での再確認
        # (d) 交差が検出されたトラック

        need_reid_track_ids = set()
        
        # (b) ロストから復帰したトラック
        need_reid_track_ids.update(recovered_track_ids)
        
        # (c) 一定間隔での再確認
        for track in next_active_tracks:
            last_verified = self.track_last_verified.get(track.track_id, 0)
            if self.frame_count - last_verified >= self.reverify_interval:
                need_reid_track_ids.add(track.track_id)
        
        # (d) 交差が検出されたトラック
        need_reid_track_ids.update(overlapping_track_ids)

        # ========================================
        # ステップ4: 必要な場合のみReID特徴量を抽出
        # ========================================
        # 新規検出のReID（未マッチ検出のみ）
        new_tracks_created = []
        if unmatched_det_indices:
            unmatched_indices = list(unmatched_det_indices)
            unmatched_boxes = boxes[unmatched_indices]
            unmatched_scores = scores[unmatched_indices]
            unmatched_classes = classes[unmatched_indices]
            unmatched_det_inds = det_inds[unmatched_indices]

            # 新規検出のみReID特徴量を抽出
            if len(unmatched_boxes) > 0:
                features = self.model.get_features(unmatched_boxes, img)
                features = np.array([f / np.linalg.norm(f) if np.linalg.norm(f) > 0 else f for f in features])

                for i, (feat, score) in enumerate(zip(features, unmatched_scores)):
                    if score > nth:
                        reg_id, sim = self._match_detection_with_registered(feat)
                        if reg_id is not None:
                            # 同じ登録IDのトラックが既に存在するか確認
                            existing_track = self._find_track_by_registered_id(reg_id)
                            if existing_track is None:
                                new_track = self._new_track(
                                    box=unmatched_boxes[i],
                                    frame_id=self.frame_count,
                                    conf=unmatched_scores[i],
                                    cls=unmatched_classes[i],
                                    det_ind=unmatched_det_inds[i],
                                    registered_id=reg_id,
                                )
                                new_tracks_created.append(new_track)
                                self.track_last_verified[reg_id] = self.frame_count

        next_active_tracks.extend(new_tracks_created)

        # 既存トラックの再確認（必要な場合のみ）
        if need_reid_track_ids:
            tracks_to_verify = [t for t in next_active_tracks if t.track_id in need_reid_track_ids]
            if tracks_to_verify:
                verify_boxes = np.array([t.bbox for t in tracks_to_verify])
                verify_features = self.model.get_features(verify_boxes, img)
                verify_features = np.array([f / np.linalg.norm(f) if np.linalg.norm(f) > 0 else f for f in verify_features])

                for track, feat in zip(tracks_to_verify, verify_features):
                    reg_id, sim = self._match_detection_with_registered(feat)
                    if reg_id is not None and reg_id == track.registered_id:
                        # 期待通りのIDにマッチ → 確認OK
                        self.track_last_verified[track.track_id] = self.frame_count
                    elif reg_id is not None and reg_id != track.registered_id:
                        # 異なるIDにマッチ → ID切り替えの可能性
                        # 既にそのIDのトラックがなければ更新
                        existing = self._find_track_by_registered_id(reg_id)
                        if existing is None:
                            track.registered_id = reg_id
                            track.track_id = reg_id
                            self.track_last_verified[reg_id] = self.frame_count
                    # マッチしない場合はそのまま継続

        # ========================================
        # ステップ5: ロストトラックの処理
        # ========================================
        unmatched_track_pool = [track_pool[idx] for idx in range(len(track_pool)) 
                                if idx not in matched_track_indices]
        self._update_lost_tracks(unmatched_track_pool)
        self.active_tracks = next_active_tracks.copy()

        # 出力の生成
        outputs = [self._format_track(track) for track in next_active_tracks]
        return np.asarray(outputs, dtype=float) if outputs else np.empty((0, 8), dtype=float)

    def _detect_overlapping_tracks(self, tracks: list[Track]) -> set[int]:
        """重なり合っているトラックのIDを検出."""
        overlapping_ids = set()
        if len(tracks) < 2:
            return overlapping_ids

        boxes = np.array([t.bbox for t in tracks])
        n = len(boxes)

        for i in range(n):
            for j in range(i + 1, n):
                iou = self._calculate_iou(boxes[i], boxes[j])
                if iou > self.overlap_iou_threshold:
                    overlapping_ids.add(tracks[i].track_id)
                    overlapping_ids.add(tracks[j].track_id)

        return overlapping_ids

    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """2つのボックス間のIoUを計算."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _dynamic_thresholds(self, scores: np.ndarray) -> tuple[float, float, float]:
        """動的閾値を計算."""
        hth = self.high_th
        nth = self.new_track_th
        mth = self.match_th_first
        if self.dynamic_tuning:
            count = len(scores[scores > self.cth])
            if count < 1:
                count = 1
            lnc = np.log10(count)
            hth = self.clamp(hth - (self.high_th_m * lnc), 0.0, 1.0)
            nth = self.clamp(nth + (self.new_track_th_m * lnc), hth, 1.0)
            mth = self.clamp(mth - (self.match_th_first_m * lnc), 0.0, 0.67)
        return hth, nth, mth

    def _purge_stale_lost_tracks(self) -> None:
        """期限切れのロストトラックを削除."""
        for track in self.lost_stracks.copy():
            if track.state == TrackState.Lost_Central:
                if self.frame_count - track.last_frame > self.central_timeout:
                    self.lost_stracks.remove(track)
            else:
                if self.frame_count - track.last_frame > self.marginal_timeout:
                    self.lost_stracks.remove(track)

    def _update_lost_tracks(self, next_lost_tracks: Iterable[Track]) -> None:
        """ロストトラックのリストを更新."""
        for track in next_lost_tracks:
            if track not in self.lost_stracks:
                self.lost_stracks.append(track)
                self._update_track_state_for_loss(track)

    def _update_track_state_for_loss(self, track: Track) -> None:
        """トラックのロスト状態を更新（中央部/周辺部）."""
        u = track.bbox[0] + (track.bbox[2] - track.bbox[0]) / 2.0
        v = track.bbox[1] + (track.bbox[3] - track.bbox[1]) / 2.0
        if (self.l_margin < u < self.r_margin) and (self.t_margin < v < self.b_margin):
            track.state = TrackState.Lost_Central
        else:
            track.state = TrackState.Lost_Marginal

    def _maybe_set_margins(self, frame_width: int | None, frame_height: int | None) -> None:
        """マージンを設定."""
        if frame_width is None or frame_height is None:
            return

        self.l_margin = 0.0
        self.r_margin = float(frame_width)
        if self.horizontal_margin is not None:
            self.l_margin = float(self.clamp(self.horizontal_margin, 0, frame_width))
            self.r_margin = float(self.clamp(frame_width - self.horizontal_margin, 0, frame_width))

        self.t_margin = 0.0
        self.b_margin = float(frame_height)
        if self.vertical_margin is not None:
            self.t_margin = float(self.clamp(self.vertical_margin, 0, frame_height))
            self.b_margin = float(self.clamp(frame_height - self.vertical_margin, 0, frame_height))

        self._margins_ready = True

    def _new_track(
        self,
        box: np.ndarray,
        frame_id: int,
        conf: float,
        cls: float,
        det_ind: int,
        registered_id: int,
    ) -> Track:
        """新規トラックを作成（登録IDを使用）."""
        track = Track(
            bbox=box,
            last_frame=frame_id,
            track_id=registered_id,  # 登録IDをトラックIDとして使用
            conf=float(conf),
            cls=int(cls),
            det_ind=int(det_ind),
            registered_id=registered_id,
        )
        return track

    @staticmethod
    def _format_track(track: Track) -> list[float]:
        """トラックを出力形式にフォーマット."""
        return [
            float(track.bbox[0]),
            float(track.bbox[1]),
            float(track.bbox[2]),
            float(track.bbox[3]),
            float(track.track_id),
            float(track.conf),
            float(track.cls),
            float(track.det_ind),
        ]

    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """値を範囲内にクランプ."""
        return max(min_value, min(value, max_value))

    @staticmethod
    def _resolve_or_default(
        value: float | None, default: float, min_value: float, max_value: float
    ) -> float:
        """値を解決し、デフォルトまたは範囲内にクランプ."""
        resolved = default if value is None else value
        return PreloadSFSORT.clamp(resolved, min_value, max_value)

    @staticmethod
    def calculate_cost(tracks: list[Track], boxes: np.ndarray, iou_only: bool = False) -> np.ndarray:
        """BBSIコスト関数を使用してアソシエーションコストを計算."""
        eps = 1e-7
        active_boxes = [track.bbox for track in tracks]
        if len(active_boxes) == 0 or boxes.size == 0:
            return np.empty((len(active_boxes), len(boxes)))

        b1_x1, b1_y1, b1_x2, b1_y2 = np.array(active_boxes).T
        b2_x1, b2_y1, b2_x2, b2_y2 = np.array(boxes).T

        h_intersection = (
            np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)
        ).clip(0)
        w_intersection = (
            np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
        ).clip(0)

        intersection = h_intersection * w_intersection

        box1_height = b1_x2 - b1_x1
        box2_height = b2_x2 - b2_x1
        box1_width = b1_y2 - b1_y1
        box2_width = b2_y2 - b2_y1

        box1_area = box1_height * box1_width
        box2_area = box2_height * box2_width

        union = box2_area + box1_area[:, None] - intersection + eps
        iou = intersection / union

        if iou_only:
            return 1.0 - iou

        # BBSI（Bounding Box Similarity Index）の計算
        centerx1 = (b1_x1 + b1_x2) / 2.0
        centery1 = (b1_y1 + b1_y2) / 2.0
        centerx2 = (b2_x1 + b2_x2) / 2.0
        centery2 = (b2_y1 + b2_y2) / 2.0
        inner_diag = np.abs(centerx1[:, None] - centerx2) + np.abs(centery1[:, None] - centery2)

        xxc1 = np.minimum(b1_x1[:, None], b2_x1)
        yyc1 = np.minimum(b1_y1[:, None], b2_y1)
        xxc2 = np.maximum(b1_x2[:, None], b2_x2)
        yyc2 = np.maximum(b1_y2[:, None], b2_y2)
        outer_diag = np.abs(xxc2 - xxc1) + np.abs(yyc2 - yyc1)

        diou = iou - (inner_diag / outer_diag)

        delta_w = np.abs(box2_width - box1_width[:, None])
        sw = w_intersection / np.abs(w_intersection + delta_w + eps)

        delta_h = np.abs(box2_height - box1_height[:, None])
        sh = h_intersection / np.abs(h_intersection + delta_h + eps)

        bbsi = diou + sh + sw
        cost = bbsi / 3.0
        return 1.0 - cost
