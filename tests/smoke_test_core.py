from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.airsim_client.client import AirSimClient
from src.commands.drone_control_state import DroneControlPhase, DroneControlStateMachine
from src.commands.gesture_command_mapper import GestureCommandMapper
from src.commands.rate_limiter import CommandRateLimiter
from src.commands.stabilizer import GestureStabilizer
from src.inference.model_loader import GestureClassifier, ModelCatalog
from src.inference.predictor import GesturePredictor
from src.utils.config_loader import load_app_config


def main() -> int:
    config = load_app_config()
    catalog = ModelCatalog(config["models"]["manifest"])
    assert {"resnet18", "mobilenetv3_small"}.issubset(set(catalog.model_names))

    mapper = GestureCommandMapper(config["commands"]["mapping"])
    assert mapper.from_gesture("like").command == "forward"
    assert mapper.safe().command == "hover"

    airsim_client = AirSimClient(config["airsim"])
    assert airsim_client.forward_speed == config["airsim"]["forward_speed_mps"]
    assert airsim_client.lateral_speed == config["airsim"]["lateral_speed_mps"]
    assert airsim_client.yaw_rate == 45.0

    stabilizer = GestureStabilizer(required_frames=3, min_confidence=0.7)
    assert not stabilizer.update("like", 0.9).stable
    assert not stabilizer.update("like", 0.9).stable
    assert stabilizer.update("like", 0.9).stable

    limiter = CommandRateLimiter(send_rate_hz=5)
    assert limiter.should_send("hover")
    assert not limiter.should_send("hover")
    assert limiter.should_send("forward")
    limiter = CommandRateLimiter(send_rate_hz=5, repeat_same_command_s=10)
    assert limiter.should_send("hover")
    assert not limiter.should_send("hover")
    assert limiter.should_send("forward")

    control = DroneControlStateMachine(motion_refresh_s=0.05)
    control.on_connected(landed=True)
    assert control.phase == DroneControlPhase.GROUNDED
    assert control.request("forward").accepted is False
    assert control.request("takeoff").accepted
    assert control.request("hover").accepted is False
    control.on_command_finished("takeoff", success=True, landed=False)
    assert control.request("forward").accepted
    assert control.request("forward").accepted is False
    time.sleep(0.06)
    assert control.request("forward").accepted
    assert control.request("land").accepted
    assert control.request("forward").accepted is False
    control.on_command_finished("land", success=True, landed=True)
    assert control.phase == DroneControlPhase.GROUNDED

    for model_name in catalog.model_names:
        classifier = GestureClassifier.load(catalog.get(model_name), preferred_device="cpu", fallback_device="cpu")
        predictor = GesturePredictor(classifier)
        dummy_roi = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_roi(dummy_roi)
        assert result.gesture in classifier.definition.labels
        assert 0.0 <= result.confidence <= 1.0
        print(f"{model_name}: {result.gesture} {result.confidence:.4f}")

    print("smoke_test_core ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
