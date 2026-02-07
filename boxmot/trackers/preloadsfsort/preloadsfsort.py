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
        """フレームを処理し、トラック情報を更新."""
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

        # ReID特徴量の抽出
        if dets.size and len(boxes) > 0:
            features = self.model.get_features(boxes, img)
            # 正規化
            features = np.array([f / np.linalg.norm(f) if np.linalg.norm(f) > 0 else f for f in features])
        else:
            features = np.empty((0, 512))  # デフォルトの特徴量次元

        # 事前登録画像とのマッチング
        det_to_registered = {}  # {det_idx: (registered_id, similarity)}
        for idx, feat in enumerate(features):
            reg_id, sim = self._match_detection_with_registered(feat)
            if reg_id is not None:
                det_to_registered[idx] = (reg_id, sim)

        # フィルタリング: 登録済み人物にマッチした検出のみを処理
        filtered_indices = list(det_to_registered.keys())
        if len(filtered_indices) == 0:
            # 登録済み人物が検出されなかった場合
            self._purge_stale_lost_tracks()
            for track in self.active_tracks:
                self.lost_stracks.append(track)
                self._update_track_state_for_loss(track)
            self.active_tracks = []
            return np.empty((0, 8), dtype=float)

        filtered_boxes = boxes[filtered_indices]
        filtered_scores = scores[filtered_indices]
        filtered_classes = classes[filtered_indices]
        filtered_det_inds = det_inds[filtered_indices]
        filtered_features = features[filtered_indices]

        # 動的閾値の計算
        hth, nth, mth = self._dynamic_thresholds(filtered_scores)

        next_active_tracks: list[Track] = []
        self._purge_stale_lost_tracks()

        track_pool = self.active_tracks + self.lost_stracks
        unmatched_tracks = np.array([], dtype=int)

        # 高信頼度検出の処理
        high_score_mask = filtered_scores > hth
        if high_score_mask.any():
            high_indices = np.where(high_score_mask)[0]
            definite_boxes = filtered_boxes[high_score_mask]
            definite_scores = filtered_scores[high_score_mask]
            definite_classes = filtered_classes[high_score_mask]
            definite_det_inds = filtered_det_inds[high_score_mask]

            if track_pool:
                cost = self.calculate_cost(track_pool, definite_boxes)
                matches, unmatched_tracks, unmatched_detections = linear_assignment(cost, mth)

                for track_idx, detection_idx in matches:
                    track = track_pool[track_idx]
                    track.update(
                        definite_boxes[detection_idx],
                        self.frame_count,
                        definite_scores[detection_idx],
                        definite_classes[detection_idx],
                        definite_det_inds[detection_idx],
                    )
                    next_active_tracks.append(track)
                    if track in self.lost_stracks:
                        self.lost_stracks.remove(track)

                for det_idx in unmatched_detections:
                    if definite_scores[det_idx] > nth:
                        # 新規トラック作成（事前登録IDを使用）
                        original_filtered_idx = high_indices[det_idx]
                        original_det_idx = filtered_indices[original_filtered_idx]
                        reg_id, _ = det_to_registered[original_det_idx]

                        # 同じ登録IDのトラックが既に存在するか確認
                        existing_track = self._find_track_by_registered_id(reg_id)
                        if existing_track is None:
                            new_track = self._new_track(
                                box=definite_boxes[det_idx],
                                frame_id=self.frame_count,
                                conf=definite_scores[det_idx],
                                cls=definite_classes[det_idx],
                                det_ind=definite_det_inds[det_idx],
                                registered_id=reg_id,
                            )
                            next_active_tracks.append(new_track)
            else:
                # トラックプールが空の場合、新規トラックを作成
                for det_idx, score in enumerate(definite_scores):
                    if score > nth:
                        original_filtered_idx = high_indices[det_idx]
                        original_det_idx = filtered_indices[original_filtered_idx]
                        reg_id, _ = det_to_registered[original_det_idx]

                        existing_track = self._find_track_by_registered_id(reg_id)
                        if existing_track is None:
                            new_track = self._new_track(
                                box=definite_boxes[det_idx],
                                frame_id=self.frame_count,
                                conf=definite_scores[det_idx],
                                cls=definite_classes[det_idx],
                                det_ind=definite_det_inds[det_idx],
                                registered_id=reg_id,
                            )
                            next_active_tracks.append(new_track)

        # 未マッチトラックの処理
        unmatched_track_pool = [track_pool[idx] for idx in unmatched_tracks] if len(unmatched_tracks) else []
        next_lost_tracks = unmatched_track_pool.copy()

        # 中間信頼度検出の処理（第2段階マッチング）
        intermediate_score_mask = np.logical_and(self.low_th < filtered_scores, filtered_scores <= hth)
        if intermediate_score_mask.any() and len(unmatched_tracks):
            possible_boxes = filtered_boxes[intermediate_score_mask]
            possible_scores = filtered_scores[intermediate_score_mask]
            possible_classes = filtered_classes[intermediate_score_mask]
            possible_det_inds = filtered_det_inds[intermediate_score_mask]

            cost = self.calculate_cost(unmatched_track_pool, possible_boxes, iou_only=True)
            matches, _, _ = linear_assignment(cost, self.match_th_second)

            for track_idx, detection_idx in matches:
                track = unmatched_track_pool[track_idx]
                track.update(
                    possible_boxes[detection_idx],
                    self.frame_count,
                    possible_scores[detection_idx],
                    possible_classes[detection_idx],
                    possible_det_inds[detection_idx],
                )
                next_active_tracks.append(track)
                if track in self.lost_stracks:
                    self.lost_stracks.remove(track)
                if track in next_lost_tracks:
                    next_lost_tracks.remove(track)

        # 検出がなかった場合
        if not (high_score_mask.any() or intermediate_score_mask.any()):
            next_lost_tracks = track_pool.copy()

        # ロストトラックの更新
        self._update_lost_tracks(next_lost_tracks)
        self.active_tracks = next_active_tracks.copy()

        # 出力の生成
        outputs = [self._format_track(track) for track in next_active_tracks]
        return np.asarray(outputs, dtype=float) if outputs else np.empty((0, 8), dtype=float)

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
