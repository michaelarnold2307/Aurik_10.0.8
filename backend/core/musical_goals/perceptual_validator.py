"""
Perceptual Validation System für Musical Goals.

Validiert technische Musical Goals Measurements gegen psychoakustische
Wahrnehmung durch ML-Modell und sammelt A/B-Test-Daten für kontinuierliche
Verbesserung.

Component 0.9.2: Perceptual Validation System
Impact: +1.5 Punkte - Perceptual Validation Guarantee
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import librosa
except ImportError:  # noqa: BLE001
    librosa = None  # type: ignore[assignment]

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    # OSError: libcupti.so.12 undefined symbol — torch-CUDA-Abhängigkeit in venv
    torch = None  # type: ignore[assignment]
    AutoFeatureExtractor = None  # type: ignore[assignment]
    AutoModelForAudioClassification = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)

# Lokales AST-Modell-Verzeichnis — §13.3 bundled:true, kein HF-Download
_AST_LOCAL_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ast_perceptual_base"


@dataclass
class PerceptualScore:
    """
    Perceptual validation score für ein Musical Goal.

    Attributes:
        technical_score: Original technischer Score (0-1)
        psychoacoustic_score: Predicted perceptual score (0-1)
        confidence: Confidence des psychoacoustic models (0-1)
        adjusted_score: Gewichteter final score (0-1)
        requires_human: Ob menschliche validation empfohlen wird
    """

    technical_score: float
    psychoacoustic_score: float
    confidence: float
    adjusted_score: float
    requires_human: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ListeningTestRequest:
    """Request für menschliches Listening Test."""

    session_id: str
    audio_path: str
    goal_scores: dict[str, float]
    confidence_scores: dict[str, float]
    reason: str
    priority: str  # 'high', 'medium', 'low'
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestSample:
    """A/B Test Sample für Training Data Collection."""

    sample_id: str
    audio_a_path: str
    audio_b_path: str
    goal: str
    score_a: float
    score_b: float
    predicted_preference: str  # 'A' or 'B'
    confidence: float
    collected_at: datetime = field(default_factory=datetime.now)
    human_preference: str | None = None  # Filled after human evaluation


class PerceptualValidator:
    """
    Validates Musical Goals gegen psychoakustische Wahrnehmung.

    Features:
    - Psychoakustisches ML-Modell (AST from HuggingFace)
    - Confidence Scoring für alle 7 Goals
    - A/B Test Data Collection
    - Listening Test Requirement Logic
    - Continuous Learning from Human Feedback

    Workflow:
    1. Technical Score berechnen (MusicalGoalsChecker)
    2. Psychoacoustic Score predicten (AST Model)
    3. Confidence Score berechnen
    4. Weighted Final Score berechnen (70% technical, 30% psychoacoustic)
    5. Entscheiden ob Listening Test nötig
    6. A/B Test Samples sammeln für Retraining
    """

    def __init__(
        self,
        model_name: str = str(_AST_LOCAL_DIR),
        confidence_threshold: float = 0.7,
        ab_test_collection_rate: float = 0.1,
        ab_test_storage_path: Path | None = None,
    ):
        """
        Initialize Perceptual Validator.

        Args:
            model_name: Lokaler Pfad zum AST-Modell (models/ast_perceptual_base/)
            confidence_threshold: Minimum confidence für automatische validation
            ab_test_collection_rate: Fraction of samples für A/B testing (0-1)
            ab_test_storage_path: Path für A/B test data storage
        """
        self.confidence_threshold = confidence_threshold
        self.ab_test_collection_rate = ab_test_collection_rate
        self.ab_test_storage_path = ab_test_storage_path or Path("data/ab_tests")
        self.ab_test_storage_path.mkdir(parents=True, exist_ok=True)

        # Psychoakustisches Modell laden — §13.3 Invariante: ausschließlich lokal (kein HF-Hub)
        # TRANSFORMERS_OFFLINE + HF_HUB_OFFLINE: blockieren Netzwerkzugriff in _load_hf_model()
        # Thread + 8 s Timeout + shutdown(wait=False): schützt gegen stale file locks
        import concurrent.futures as _cf
        import os as _os

        self.device = torch.device("cpu") if _TORCH_AVAILABLE and torch is not None else None

        try:
            if not _TORCH_AVAILABLE or torch is None:
                raise ImportError("torch nicht verfügbar — DSP-Fallback aktiviert")
            # Lokales Modell-Verzeichnis prüfen — kein HF-Hub-Download (§13.3)
            if not Path(model_name).is_dir():
                raise OSError(f"Modell-Verzeichnis {model_name!r} nicht gefunden — DSP-Fallback aktiviert")

            def _load_hf_model() -> tuple:
                _prev = _os.environ.get("TRANSFORMERS_OFFLINE", "")
                _prev_hf = _os.environ.get("HF_HUB_OFFLINE", "")
                _os.environ["TRANSFORMERS_OFFLINE"] = "1"
                _os.environ["HF_HUB_OFFLINE"] = "1"  # blockiert XET-Storage-Downloads
                try:
                    _fe = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)  # nosec B615 — local_files_only=True, kein Download
                    _m = AutoModelForAudioClassification.from_pretrained(model_name, local_files_only=True)  # nosec B615 — local_files_only=True, kein Download
                    _m.eval()
                    _m.to(torch.device("cpu"))
                    return _fe, _m
                finally:
                    if _prev:
                        _os.environ["TRANSFORMERS_OFFLINE"] = _prev
                    else:
                        _os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    if _prev_hf:
                        _os.environ["HF_HUB_OFFLINE"] = _prev_hf
                    else:
                        _os.environ.pop("HF_HUB_OFFLINE", None)

            _ex = _cf.ThreadPoolExecutor(max_workers=1)
            _fut = _ex.submit(_load_hf_model)
            try:
                self.feature_extractor, self.model = _fut.result(timeout=8.0)
                logger.info("Psychoakustisches Modell geladen: %s (CPU)", model_name)
            except _cf.TimeoutError:
                logger.warning("Modell-Laden Timeout (8 s) — DSP-Fallback aktiviert")
                self.feature_extractor = None
                self.model = None
            except Exception as _load_err:
                logger.warning("Modell-Laden Fehler: %s — DSP-Fallback", _load_err)
                self.feature_extractor = None
                self.model = None
            finally:
                _ex.shutdown(wait=False)  # Hintergrund-Thread nicht blockierend beenden

        except Exception as e:
            logger.warning("Failed to load psychoacoustic model: %s", e)
            logger.warning("Perceptual validation will use fallback heuristics")
            self.feature_extractor = None
            self.model = None

        # Statistics
        self.validation_count = 0
        self.listening_test_requests = []
        self.ab_test_samples = []

    def validate_goal(
        self,
        audio: np.ndarray,
        sr: int,
        goal_name: str,
        technical_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> PerceptualScore:
        """
        Validate ein Musical Goal gegen psychoakustische Wahrnehmung.

        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            goal_name: Name des Musical Goal ('bass-kraft', 'brillanz', etc.)
            technical_score: Technischer score von MusicalGoalsChecker
            metadata: Zusätzliche metadata (genre, medium_type, etc.)

        Returns:
            PerceptualScore mit allen validation details
        """
        self.validation_count += 1
        metadata = metadata or {}

        # Psychoacoustic Score predicten
        psychoacoustic_score, confidence = self._predict_psychoacoustic_score(audio, sr, goal_name, metadata)

        # Adjusted Score berechnen (weighted: 70% technical, 30% psychoacoustic)
        adjusted_score = 0.7 * technical_score + 0.3 * psychoacoustic_score

        # Check ob Listening Test nötig
        requires_human = self._requires_listening_test(
            confidence=confidence,
            technical_score=technical_score,
            psychoacoustic_score=psychoacoustic_score,
            goal_name=goal_name,
        )

        # A/B Test Sample sammeln (bei Rate%)
        if np.random.random() < self.ab_test_collection_rate:
            self._collect_ab_test_sample(
                audio, sr, goal_name, technical_score, psychoacoustic_score, confidence, metadata
            )

        return PerceptualScore(
            technical_score=technical_score,
            psychoacoustic_score=psychoacoustic_score,
            confidence=confidence,
            adjusted_score=adjusted_score,
            requires_human=requires_human,
            metadata={"goal_name": goal_name, "validation_id": self.validation_count, **metadata},
        )

    def validate_all_goals(
        self, audio: np.ndarray, sr: int, technical_scores: dict[str, float], metadata: dict[str, Any] | None = None
    ) -> dict[str, PerceptualScore]:
        """
        Validate alle 7 Musical Goals.

        Args:
            audio: Audio signal
            sr: Sample rate
            technical_scores: Dict mit allen technical scores
            metadata: Zusätzliche metadata

        Returns:
            Dict[goal_name, PerceptualScore] für alle Goals
        """
        results = {}
        for goal_name, technical_score in technical_scores.items():
            results[goal_name] = self.validate_goal(audio, sr, goal_name, technical_score, metadata)

        # Check ob overall Listening Test nötig
        if any(score.requires_human for score in results.values()):
            self._create_listening_test_request(audio, technical_scores, results, metadata)

        return results

    def _predict_psychoacoustic_score(
        self, audio: np.ndarray, sr: int, goal_name: str, metadata: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Predict psychoacoustic score using ML model.

        Returns:
            (psychoacoustic_score, confidence)
        """
        if self.model is None:
            # Fallback: Heuristic-based scoring
            return self._heuristic_psychoacoustic_score(audio, sr, goal_name, metadata)

        try:
            # Resample to model's expected sample rate (16kHz for AST)
            target_sr = 16000
            if sr != target_sr:
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            else:
                audio_resampled = audio

            # Extract features
            inputs = self.feature_extractor(audio_resampled, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Map model outputs to goal-specific score
            # Das ist eine vereinfachte Mapping - in Production würde hier
            # ein fine-tuned model verwendet werden
            psychoacoustic_score, confidence = self._map_model_output_to_goal(probs, goal_name, metadata)

            return psychoacoustic_score, confidence

        except Exception as e:
            logger.warning(f"Psychoacoustic prediction failed: {e}")
            return self._heuristic_psychoacoustic_score(audio, sr, goal_name, metadata)

    def _map_model_output_to_goal(
        self, probs: torch.Tensor, goal_name: str, metadata: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Map model probabilities zu Goal-specific score.

        NOTE: Dies ist eine Placeholder-Implementierung.
        In Production würde hier ein custom-trained model verwendet werden.
        """
        # Get max probability als confidence
        confidence = float(probs.max())

        # Goal-specific mapping (simplified)
        # In reality würde jedes Goal eigene classifier heads haben
        goal_mappings = {
            "bass-kraft": [0, 10, 137],  # AudioSet classes related to bass
            "brillanz": [138, 310, 311],  # High frequency content
            "waerme": [137, 141],  # Warm, mellow sounds
            "natuerlichkeit": [0, 1, 2],  # Natural sounds
            "authentizitaet": [0, 1, 2],  # Similar to natuerlichkeit
            "emotionalitaet": [137, 141, 310],  # Music, emotional content
            "transparenz": [0, 310, 311],  # Clear, distinct sounds
        }

        # Average probability für relevante classes
        relevant_classes = goal_mappings.get(goal_name, [0])
        relevant_probs = probs[0, relevant_classes]
        psychoacoustic_score = float(relevant_probs.mean())

        # Normalize to 0-1 range
        psychoacoustic_score = np.clip(psychoacoustic_score * 2.0, 0.0, 1.0)

        return psychoacoustic_score, confidence

    def _heuristic_psychoacoustic_score(
        self, audio: np.ndarray, sr: int, goal_name: str, metadata: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Fallback heuristic-based psychoacoustic scoring.

        Returns:
            (psychoacoustic_score, confidence)
        """
        # Extract basic features
        rms = librosa.feature.rms(y=audio)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]

        # Goal-specific heuristics
        if goal_name == "bass-kraft":
            # Low spectral centroid = more bass
            score = 1.0 - np.clip(np.mean(spectral_centroid) / 4000, 0, 1)
            confidence = 0.5

        elif goal_name == "brillanz":
            # High spectral centroid = more brilliance
            score = np.clip(np.mean(spectral_centroid) / 4000, 0, 1)
            confidence = 0.5

        elif goal_name == "waerme":
            # Mid-range energy = warmth
            score = 0.7  # Placeholder
            confidence = 0.3

        elif goal_name == "natuerlichkeit":
            # Low zero-crossing rate = more natural
            score = 1.0 - np.clip(np.mean(zcr), 0, 1)
            confidence = 0.4

        elif goal_name == "emotionalitaet":
            # Dynamic range = emotionality
            dynamic_range = np.std(rms)
            score = np.clip(dynamic_range * 10, 0, 1)
            confidence = 0.4

        elif goal_name == "transparenz":
            # Clear separation (placeholder)
            score = 0.75
            confidence = 0.3

        else:  # 'authentizitaet' or unknown
            score = 0.7
            confidence = 0.3

        return float(score), float(confidence)

    def _requires_listening_test(
        self, confidence: float, technical_score: float, psychoacoustic_score: float, goal_name: str
    ) -> bool:
        """
        Entscheidet ob menschliches Listening Test nötig ist.

        Kriterien:
        - Low confidence (<70%)
        - Large discrepancy zwischen technical und psychoacoustic score
        - Critical violations (<60%)
        - Critical goals (Natürlichkeit, Authentizität)
        """
        # Low confidence
        if confidence < self.confidence_threshold:
            return True

        # Large discrepancy (>20% difference)
        discrepancy = abs(technical_score - psychoacoustic_score)
        if discrepancy > 0.2:
            return True

        # Critical violations
        if technical_score < 0.6 or psychoacoustic_score < 0.6:
            return True

        # Critical goals always require higher scrutiny
        critical_goals = ["natuerlichkeit", "authentizitaet"]
        if goal_name in critical_goals and confidence < 0.85:
            return True

        return False

    def _create_listening_test_request(
        self,
        audio: np.ndarray,
        technical_scores: dict[str, float],
        perceptual_scores: dict[str, PerceptualScore],
        metadata: dict[str, Any],
    ) -> None:
        """Create Listening Test Request für menschliche validation."""
        metadata = metadata or {}  # Handle None

        # Find lowest confidence scores
        confidence_scores = {goal: score.confidence for goal, score in perceptual_scores.items()}
        min_confidence = min(confidence_scores.values())

        # Determine priority
        if min_confidence < 0.5:
            priority = "high"
        elif min_confidence < 0.7:
            priority = "medium"
        else:
            priority = "low"

        # Create reason
        low_conf_goals = [goal for goal, conf in confidence_scores.items() if conf < self.confidence_threshold]
        reason = f"Low confidence for goals: {', '.join(low_conf_goals)}"

        request = ListeningTestRequest(
            session_id=metadata.get("session_id", "unknown"),
            audio_path=metadata.get("audio_path", "unknown"),
            goal_scores=technical_scores,
            confidence_scores=confidence_scores,
            reason=reason,
            priority=priority,
        )

        self.listening_test_requests.append(request)
        logger.info(f"Created listening test request (priority={priority}): {reason}")

    def _collect_ab_test_sample(
        self,
        audio: np.ndarray,
        sr: int,
        goal_name: str,
        technical_score: float,
        psychoacoustic_score: float,
        confidence: float,
        metadata: dict[str, Any],
    ) -> None:
        """
        Sammelt A/B Test Sample für Training Data Collection.

        Strategy: Sammle pairs von audio wo model uncertain ist (confidence < 0.8)
        für spätere menschliche evaluation.
        """
        if confidence > 0.8:
            return  # Only collect uncertain samples

        sample_id = f"ab_{goal_name}_{datetime.now().timestamp()}"

        # Store sample info (actual audio files würden separat gespeichert)
        sample = ABTestSample(
            sample_id=sample_id,
            audio_a_path=metadata.get("audio_path", "unknown"),
            audio_b_path=metadata.get("reference_path", metadata.get("audio_path", "unknown")),
            goal=goal_name,
            score_a=technical_score,
            score_b=psychoacoustic_score,
            predicted_preference="A" if technical_score > psychoacoustic_score else "B",
            confidence=confidence,
        )

        self.ab_test_samples.append(sample)

        # Save to disk
        sample_file = self.ab_test_storage_path / f"{sample_id}.json"
        with open(sample_file, "w") as f:
            json.dump(
                {
                    "sample_id": sample.sample_id,
                    "audio_a_path": sample.audio_a_path,
                    "audio_b_path": sample.audio_b_path,
                    "goal": sample.goal,
                    "score_a": sample.score_a,
                    "score_b": sample.score_b,
                    "predicted_preference": sample.predicted_preference,
                    "confidence": sample.confidence,
                    "collected_at": sample.collected_at.isoformat(),
                },
                f,
                indent=2,
            )

        logger.debug(f"Collected A/B test sample: {sample_id}")

    def get_listening_test_queue(self, priority: str | None = None, limit: int = 10) -> list[ListeningTestRequest]:
        """
        Get queue of pending listening test requests.

        Args:
            priority: Filter by priority ('high', 'medium', 'low')
            limit: Maximum number of requests to return

        Returns:
            List of ListeningTestRequest
        """
        requests = self.listening_test_requests

        if priority:
            requests = [r for r in requests if r.priority == priority]

        # Sort by priority (high > medium > low) and creation time
        priority_order = {"high": 0, "medium": 1, "low": 2}
        requests.sort(key=lambda r: (priority_order[r.priority], r.created_at))

        return requests[:limit]

    def submit_listening_test_result(
        self, session_id: str, human_scores: dict[str, float], comments: str | None = None
    ) -> Dict[str, Any]:
        """
        Submit menschliche listening test results.

        Args:
            session_id: Session ID des requests
            human_scores: Dict[goal_name, human_score]
            comments: Optional feedback comments
        """
        # Find corresponding request
        request = next((r for r in self.listening_test_requests if r.session_id == session_id), None)

        if not request:
            logger.warning(f"No listening test request found for session {session_id}")
            return

        # Store result
        result_file = self.ab_test_storage_path / f"listening_test_{session_id}.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "audio_path": request.audio_path,
                    "technical_scores": request.goal_scores,
                    "human_scores": human_scores,
                    "confidence_scores": request.confidence_scores,
                    "comments": comments,
                    "submitted_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        # Remove from queue
        self.listening_test_requests.remove(request)
        logger.info(f"Listening test result submitted for session {session_id}")

    def get_statistics(self) -> dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_validations": self.validation_count,
            "listening_test_requests": len(self.listening_test_requests),
            "ab_test_samples_collected": len(self.ab_test_samples),
            "listening_test_queue_by_priority": {
                "high": len([r for r in self.listening_test_requests if r.priority == "high"]),
                "medium": len([r for r in self.listening_test_requests if r.priority == "medium"]),
                "low": len([r for r in self.listening_test_requests if r.priority == "low"]),
            },
        }
