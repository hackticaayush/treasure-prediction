"""
predictor.py – Clean 2-class prediction. No bonus classes.
Win = true_val in {pred1_cls, pred2_cls} only. No bonus inflation.
"""

import json
import numpy as np
import joblib
import warnings
import copy
from collections import deque
warnings.filterwarnings("ignore")

# Configuration
WINDOW_SIZE  = 16
MODEL_PATH   = "best_model.pkl"
TOP_K        = 2   # Only 2 predictions now — no bonus
HARD_SKIP_CLASSES = {2, 3, 4, 7, 8}
BASE_THRESHOLD = {1: 23.0, 5: 10.0, 6: 13.0}
ADAPT_EVERY          = 30
WARMUP_ROUNDS        = 20
PLAY_RATE_WINDOW     = 60
MIN_PLAY_RATE_TARGET = 0.25
PLAY_BOOST_STEP      = 2.0
MIN_THRESHOLD        = 10.0
MAX_THRESHOLD        = 50.0
STREAK_RELAX_AFTER   = 2
THRESHOLD_RELAX_STEP = 1.5
STREAK_PATTERN_START     = 4
ENTROPY_SKIP_THRESH      = 1.90
MISMATCH_SKIP_RATIO      = 0.80
MISMATCH_WINDOW          = 6
MAX_PATTERN_SKIPS_IN_ROW = 3
STREAK_COOLDOWN_MAP = {6: 1, 8: 2, 10: 4}
POST_COOLDOWN_ROUNDS  = 5
POST_COOLDOWN_RELIEF  = 4.0

CLASS_NAMES  = {1: "Purple", 2: "10 Times", 3: "25 Times", 4: "15 Times",
                5: "Yellow", 6: "Light Green", 7: "50 Times", 8: "Dark Green"}
CLASS_COLORS = {1: "#9b59b6", 2: "#e74c3c", 3: "#e67e22", 4: "#f39c12",
                5: "#f1c40f", 6: "#2ecc71", 7: "#1abc9c", 8: "#27ae60"}


def create_features(past):
    features = []
    counts = np.bincount(past, minlength=9)[1:].astype(float)
    freq   = (counts + 1) / (len(past) + 8)
    features.extend(freq)
    last_val = past[-1]
    features.extend([1 if i == last_val else 0 for i in range(1, 9)])
    if len(past) >= 2:
        prev_val = past[-2]
        features.extend([1 if i == prev_val else 0 for i in range(1, 9)])
    features.append(int(past[-1] == past[-2]) if len(past) >= 2 else 0)
    weights = np.exp(np.linspace(-1, 0, len(past)))
    weighted_counts = np.zeros(8)
    for i, val in enumerate(past):
        weighted_counts[val - 1] += weights[i]
    weighted_counts /= weighted_counts.sum()
    features.extend(weighted_counts)
    return features


def shannon_entropy(probs):
    p = np.array(probs)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


class AdaptiveStrategy:
    def __init__(self):
        self.thresholds = dict(BASE_THRESHOLD)
        self.played_count = 0
        self.last_adapt_at = 0
        self.class_history = {c: deque(maxlen=60) for c in range(1, 9)}
        self.recent_plays = deque(maxlen=PLAY_RATE_WINDOW)
        self.streak_buffer = deque(maxlen=MISMATCH_WINDOW)
        self.consecutive_pat_skips = 0

    def record_play(self, cls, true_val, won):
        self.class_history[cls].append(int(won))
        self.recent_plays.append(True)
        self.streak_buffer.append((cls, true_val, won))
        self.played_count += 1
        self.consecutive_pat_skips = 0

    def record_skip(self, pattern_skip=False):
        self.recent_plays.append(False)
        if pattern_skip:
            self.consecutive_pat_skips += 1
        else:
            self.consecutive_pat_skips = 0

    def play_rate(self):
        return sum(self.recent_plays) / len(self.recent_plays) if self.recent_plays else 1.0

    def is_warmup(self):
        return self.played_count < WARMUP_ROUNDS

    def get_threshold(self, cls, loss_streak):
        base = self.thresholds.get(cls, 99.0)
        if loss_streak >= STREAK_RELAX_AFTER:
            relax = THRESHOLD_RELAX_STEP * (loss_streak - STREAK_RELAX_AFTER + 1)
            base = max(MIN_THRESHOLD, base - relax)
        return base

    def should_skip_streak_pattern(self, loss_streak, probs):
        if loss_streak < STREAK_PATTERN_START:
            return False, None
        if self.consecutive_pat_skips >= MAX_PATTERN_SKIPS_IN_ROW:
            return False, None
        entropy = shannon_entropy(probs)
        if entropy >= ENTROPY_SKIP_THRESH:
            return True, f"High entropy ({entropy:.3f})"
        if len(self.streak_buffer) >= 3:
            recent = list(self.streak_buffer)[-min(loss_streak, MISMATCH_WINDOW):]
            bad = sum(1 for (_, true_v, won) in recent if not won and true_v in HARD_SKIP_CLASSES)
            if len(recent) > 0 and bad / len(recent) >= MISMATCH_SKIP_RATIO:
                return True, "Mismatch"
        return False, None

    def adapt(self):
        if self.played_count - self.last_adapt_at < ADAPT_EVERY or self.is_warmup():
            return
        self.last_adapt_at = self.played_count
        for cls in BASE_THRESHOLD:
            history = list(self.class_history[cls])
            if len(history) < 8:
                continue
            win_rate = np.mean(history)
            old = self.thresholds[cls]
            if win_rate >= 0.70:
                self.thresholds[cls] = max(MIN_THRESHOLD, old - 2.0)
            elif win_rate >= 0.50:
                self.thresholds[cls] = max(MIN_THRESHOLD, old - 1.0)
            elif win_rate < 0.40:
                self.thresholds[cls] = min(MAX_THRESHOLD, old + 1.5)
        pr = self.play_rate()
        if pr < MIN_PLAY_RATE_TARGET:
            boost = PLAY_BOOST_STEP + (MIN_PLAY_RATE_TARGET - pr) * 15
            for cls in BASE_THRESHOLD:
                self.thresholds[cls] = max(MIN_THRESHOLD, self.thresholds[cls] - boost)


class LivePredictor:
    def __init__(self):
        self.ensemble = None
        self.classes = None
        self.live_loss_streak = 0
        self.cooldown_left = 0
        self.post_cooldown = 0
        self.strategy = AdaptiveStrategy()
        self.history = []
        self.last_round_seen = None
        self._load_model()

    def _load_model(self):
        try:
            self.ensemble = joblib.load(MODEL_PATH)
            self.classes = self.ensemble.classes_
            print(f"[Predictor] Model loaded from '{MODEL_PATH}'")
        except Exception as e:
            print(f"[Predictor] WARNING: Model load failed — {e}")

    def _load_rounds(self):
        with open("round_data.json", "r") as f:
            return sorted(json.load(f), key=lambda x: x["round"])

    def _process_round(self, rounds, idx):
        """Process exactly one round to update internal state. Pure 2-class win logic."""
        round_data = rounds[idx]
        true_val = round_data["reward_index"]
        past = np.array([r["reward_index"] for r in rounds[idx - WINDOW_SIZE:idx]])
        features = np.array(create_features(past)).reshape(1, -1)
        probs = self.ensemble.predict_proba(features)[0]

        top2_idx = np.argsort(probs)[-TOP_K:][::-1]
        top2_classes = self.classes[top2_idx]
        top2_confs = probs[top2_idx] * 100

        pred1_cls = int(top2_classes[0])
        pred2_cls = int(top2_classes[1])
        conf1 = float(top2_confs[0])

        self.strategy.adapt()

        skipped = False
        won = None

        if pred1_cls in HARD_SKIP_CLASSES:
            self.strategy.record_skip()
            skipped = True
        elif self.cooldown_left > 0:
            self.cooldown_left -= 1
            if self.cooldown_left == 0:
                self.post_cooldown = POST_COOLDOWN_ROUNDS
            self.strategy.record_skip()
            skipped = True
        else:
            skip_p, _ = self.strategy.should_skip_streak_pattern(self.live_loss_streak, probs)
            if skip_p:
                self.strategy.record_skip(True)
                skipped = True
            else:
                relief = POST_COOLDOWN_RELIEF if self.post_cooldown > 0 else 0.0
                if self.post_cooldown > 0:
                    self.post_cooldown -= 1
                thresh = self.strategy.get_threshold(pred1_cls, self.live_loss_streak) - relief
                if self.strategy.is_warmup():
                    thresh = min(thresh, 12.0)

                if conf1 < thresh:
                    self.strategy.record_skip()
                    skipped = True
                else:
                    # FIXED: Strict 2-class win only. No bonus. No inflation.
                    win_classes = {pred1_cls, pred2_cls}
                    won = true_val in win_classes
                    # Record win/loss against primary class for clean strategy learning
                    self.strategy.record_play(pred1_cls, true_val, won)
                    if won:
                        self.live_loss_streak = 0
                    else:
                        self.live_loss_streak += 1
                        if self.live_loss_streak in STREAK_COOLDOWN_MAP:
                            self.cooldown_left = STREAK_COOLDOWN_MAP[self.live_loss_streak]
                    skipped = False

        self.history.append({
            "round": round_data["round"],
            "true": true_val,
            "won": won,
            "skipped": skipped,
            "pred1": pred1_cls,
            "pred2": pred2_cls,
        })

    def sync(self, rounds):
        """Differential update: only process rounds not yet seen."""
        if not self.last_round_seen:
            start_idx = WINDOW_SIZE
        else:
            start_idx = next(
                (i + 1 for i, r in enumerate(rounds) if r["round"] == self.last_round_seen),
                WINDOW_SIZE
            )
        for i in range(start_idx, len(rounds)):
            self._process_round(rounds, i)
        if rounds:
            self.last_round_seen = rounds[-1]["round"]

    def predict_next(self, provided_data=None):
        if self.ensemble is None:
            return None
        rounds = provided_data if provided_data is not None else self._load_rounds()
        if len(rounds) < WINDOW_SIZE + 1:
            return None

        if provided_data is None:
            self.sync(rounds)

        values = [r["reward_index"] for r in rounds]
        past = np.array(values[-WINDOW_SIZE:])
        features = np.array(create_features(past)).reshape(1, -1)
        probs = self.ensemble.predict_proba(features)[0]

        top2_idx = np.argsort(probs)[-TOP_K:][::-1]
        top2_classes = self.classes[top2_idx]
        top2_confs = probs[top2_idx] * 100

        p1, p2 = int(top2_classes[0]), int(top2_classes[1])
        c1, c2 = float(top2_confs[0]), float(top2_confs[1])

        action = "PLAY"
        skip_reason = None

        if p1 in HARD_SKIP_CLASSES:
            action = "SKIP"
            skip_reason = "Bad Class"
        elif self.cooldown_left > 0:
            action = "SKIP"
            skip_reason = f"Cooldown ({self.cooldown_left})"
        else:
            skip_p, reason = self.strategy.should_skip_streak_pattern(self.live_loss_streak, probs)
            if skip_p:
                action = "SKIP"
                skip_reason = reason
            else:
                relief = POST_COOLDOWN_RELIEF if self.post_cooldown > 0 else 0.0
                thresh = self.strategy.get_threshold(p1, self.live_loss_streak) - relief
                if self.strategy.is_warmup():
                    thresh = min(thresh, 12.0)
                if c1 < thresh:
                    action = "SKIP"
                    skip_reason = f"Low Conf ({c1:.1f}%)"

        # FIXED: win_classes is strictly {p1, p2} — no bonus
        win_classes = sorted([p1, p2])

        return {
            "next_round":   rounds[-1]["round"] + 1,
            "latest_round": rounds[-1]["round"],
            "pred1_cls":    p1,
            "pred1_conf":   round(c1, 2),
            "pred2_cls":    p2,
            "pred2_conf":   round(c2, 2),
            "win_classes":  win_classes,
            "entropy":      round(shannon_entropy(probs), 4),
            "action":       action,
            "skip_reason":  skip_reason,
            "loss_streak":  self.live_loss_streak,
            "all_probs":    {int(self.classes[i]): round(float(probs[i]) * 100, 2)
                             for i in range(len(self.classes))},
        }

    def get_recent_history(self, n=30):
        return [h for h in self.history if not h["skipped"]][-n:]

    def get_stats(self):
        played = [h for h in self.history if not h["skipped"]]
        won = sum(1 for h in played if h["won"])
        skipped = len(self.history) - len(played)
        return {
            "total":       len(self.history),
            "played":      len(played),
            "skipped":     skipped,
            "won":         won,
            "lost":        len(played) - won,
            "win_pct":     round(won / len(played) * 100, 2) if played else 0,
            "loss_streak": self.live_loss_streak,
            "is_warmup":   self.strategy.is_warmup(),
            "play_rate":   round(self.strategy.play_rate() * 100, 1),
        }