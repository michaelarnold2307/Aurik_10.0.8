"""
Tests für Musical Goals Monitor API (Component 0.9.5 - Real-Time Visualization)

Testet WebSocket-basiertes Real-Time Monitoring System für Musical Goals.
"""

# Import from backend
import sys
import time
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import pytest

sys.path.insert(0, "/mnt/1846D15B46D139E8/Aurik_Standalone")
from backend.api.musical_goals_monitor_api import ConnectionManager, GoalsSnapshot, GoalUpdate, MusicalGoalsMonitorAPI

# Use anyio for async tests (already installed)
pytestmark = pytest.mark.anyio


class TestConnectionManager:
    """Test WebSocket connection management"""

    def test_connection_manager_init(self):
        """Test ConnectionManager initialization"""
        manager = ConnectionManager()
        assert len(manager.active_connections) == 0
        assert len(manager.session_connections) == 0

    async def test_connect_websocket(self):
        """Test WebSocket connection"""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws, "test-session")

        assert mock_ws in manager.active_connections
        assert "test-session" in manager.session_connections
        assert mock_ws in manager.session_connections["test-session"]
        mock_ws.accept.assert_called_once()

    def test_disconnect_websocket(self):
        """Test WebSocket disconnection"""
        manager = ConnectionManager()
        mock_ws = Mock(spec=WebSocket)

        # Setup connection
        manager.active_connections.append(mock_ws)
        manager.session_connections["test-session"] = [mock_ws]

        # Disconnect
        manager.disconnect(mock_ws, "test-session")

        assert mock_ws not in manager.active_connections
        assert mock_ws not in manager.session_connections.get("test-session", [])

    async def test_broadcast_to_all(self):
        """Test broadcasting message to all connections"""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        manager.active_connections = [mock_ws1, mock_ws2]

        message = {"type": "test", "data": "hello"}
        await manager.broadcast(message)

        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_called_once_with(message)

    async def test_send_to_specific_session(self):
        """Test sending message to specific session"""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        manager.session_connections["session1"] = [mock_ws1]
        manager.session_connections["session2"] = [mock_ws2]

        message = {"type": "test", "data": "session1-only"}
        await manager.send_to_session("session1", message)

        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_not_called()

    async def test_broadcast_handles_disconnected_clients(self):
        """Test broadcast continues even if one client disconnects"""
        manager = ConnectionManager()
        mock_ws_good = AsyncMock(spec=WebSocket)
        mock_ws_bad = AsyncMock(spec=WebSocket)
        mock_ws_bad.send_json.side_effect = Exception("Disconnected")

        manager.active_connections = [mock_ws_good, mock_ws_bad]

        message = {"type": "test"}
        await manager.broadcast(message)

        # Should still call both, but handle exception
        mock_ws_good.send_json.assert_called_once()
        mock_ws_bad.send_json.assert_called_once()


class TestGoalDataClasses:
    """Test goal data structures"""

    def test_goal_update_creation(self):
        """Test GoalUpdate dataclass"""
        ts = time.time()
        update = GoalUpdate(
            timestamp=ts,
            session_id="test-session",
            step_name="noise_reduction",
            goal_name="bass_kraft",
            score=0.87,
            threshold=0.85,
            violated=False,
        )

        assert update.session_id == "test-session"
        assert update.goal_name == "bass_kraft"
        assert update.score == 0.87
        assert update.violated is False

    def test_goals_snapshot_creation(self):
        """Test GoalsSnapshot dataclass"""
        ts = time.time()
        goals = {"bass_kraft": 0.87, "brillanz": 0.91}
        thresholds = {"bass_kraft": 0.85, "brillanz": 0.85}

        snapshot = GoalsSnapshot(
            timestamp=ts,
            session_id="test-session",
            step_name="noise_reduction",
            goals=goals,
            thresholds=thresholds,
            violations=[],
        )

        assert snapshot.session_id == "test-session"
        assert snapshot.goals["bass_kraft"] == 0.87
        assert len(snapshot.violations) == 0

    def test_snapshot_with_violations(self):
        """Test snapshot with violations detected"""
        ts = time.time()
        goals = {"bass_kraft": 0.82, "brillanz": 0.91}
        thresholds = {"bass_kraft": 0.85, "brillanz": 0.85}
        violations = ["bass_kraft"]

        snapshot = GoalsSnapshot(
            timestamp=ts, session_id="test", step_name="test", goals=goals, thresholds=thresholds, violations=violations
        )

        assert "bass_kraft" in snapshot.violations
        assert "brillanz" not in snapshot.violations


class TestMusicalGoalsMonitorAPI:
    """Test Musical Goals Monitor API"""

    def test_api_initialization(self):
        """Test API initialization"""
        api = MusicalGoalsMonitorAPI()

        assert api.connection_manager is not None
        assert len(api.history) == 0
        assert len(api.custom_thresholds) == 0

    def test_create_app(self):
        """Test FastAPI app creation"""
        api = MusicalGoalsMonitorAPI()
        app = api.create_app()

        assert app is not None
        # Check routes exist
        routes = [route.path for route in app.routes]
        assert "/ws/{session_id}" in routes
        assert "/api/goals/history/{session_id}" in routes
        assert "/api/goals/current/{session_id}" in routes

    async def test_update_goals(self):
        """Test goals update and broadcast"""
        api = MusicalGoalsMonitorAPI()

        goals = {"bass_kraft": 0.87, "brillanz": 0.91}
        thresholds = {"bass_kraft": 0.85, "brillanz": 0.85}

        # Mock connection manager
        api.connection_manager.send_to_session = AsyncMock()

        await api.update_goals(
            session_id="test-session", step_name="noise_reduction", goals=goals, thresholds=thresholds
        )

        # Check history updated
        assert "test-session" in api.history
        assert len(api.history["test-session"]) == 1

        # Check broadcast called
        api.connection_manager.send_to_session.assert_called_once()

    async def test_update_goals_detects_violations(self):
        """Test automatic violation detection"""
        api = MusicalGoalsMonitorAPI()

        goals = {"bass_kraft": 0.82, "brillanz": 0.91}  # Below threshold  # Above threshold
        thresholds = {"bass_kraft": 0.85, "brillanz": 0.85}

        api.connection_manager.send_to_session = AsyncMock()

        await api.update_goals(session_id="test-session", step_name="test", goals=goals, thresholds=thresholds)

        # Get snapshot from history
        snapshot = api.history["test-session"][-1]

        # Check violations detected
        assert "bass_kraft" in snapshot.violations
        assert "brillanz" not in snapshot.violations

    async def test_history_size_limit(self):
        """Test history is limited to 1000 entries per session"""
        api = MusicalGoalsMonitorAPI()
        api.connection_manager.send_to_session = AsyncMock()

        # Add 1100 updates
        for i in range(1100):
            await api.update_goals(
                session_id="test-session",
                step_name=f"step-{i}",
                goals={"bass_kraft": 0.85},
                thresholds={"bass_kraft": 0.85},
            )

        # Should be capped at 1000
        assert len(api.history["test-session"]) == 1000

    def test_set_custom_thresholds(self):
        """Test custom threshold configuration"""
        api = MusicalGoalsMonitorAPI()

        custom_thresholds = {"bass_kraft": 0.90, "brillanz": 0.88}

        # Directly set custom thresholds
        api.custom_thresholds["test-session"] = custom_thresholds

        assert "test-session" in api.custom_thresholds
        assert api.custom_thresholds["test-session"]["bass_kraft"] == 0.90

    async def test_custom_thresholds_used_in_update(self):
        """Test custom thresholds override defaults"""
        api = MusicalGoalsMonitorAPI()
        api.connection_manager.send_to_session = AsyncMock()

        # Set custom thresholds directly
        custom = {"bass_kraft": 0.90}
        api.custom_thresholds["test-session"] = custom

        # Update with score that would pass default (0.85) but not custom (0.90)
        goals = {"bass_kraft": 0.87}
        default_thresholds = {"bass_kraft": 0.85}

        await api.update_goals(session_id="test-session", step_name="test", goals=goals, thresholds=default_thresholds)

        # Should use custom threshold and detect violation
        snapshot = api.history["test-session"][-1]
        assert snapshot.thresholds["bass_kraft"] == 0.90
        assert "bass_kraft" in snapshot.violations


class TestRESTEndpoints:
    """Test REST API endpoints"""

    def setup_method(self):
        """Setup test client"""
        api = MusicalGoalsMonitorAPI()
        self.app = api.create_app()
        self.client = TestClient(self.app)
        self.api = api

    def test_get_sessions_empty(self):
        """Test get sessions when none exist"""
        response = self.client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 0

    async def test_get_history(self):
        """Test get history endpoint"""
        # Add some history
        self.api.connection_manager.send_to_session = AsyncMock()
        await self.api.update_goals(
            session_id="test-session", step_name="step1", goals={"bass_kraft": 0.85}, thresholds={"bass_kraft": 0.85}
        )

        response = self.client.get("/api/goals/history/test-session")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert len(data["history"]) >= 1

    async def test_get_current_goals(self):
        """Test get current goals endpoint"""
        # Add some history
        self.api.connection_manager.send_to_session = AsyncMock()
        await self.api.update_goals(
            session_id="test-session",
            step_name="latest-step",
            goals={"bass_kraft": 0.87},
            thresholds={"bass_kraft": 0.85},
        )

        response = self.client.get("/api/goals/current/test-session")
        assert response.status_code == 200
        data = response.json()
        assert data["step_name"] == "latest-step"
        assert data["goals"]["bass_kraft"] == 0.87

    def test_get_current_goals_no_history(self):
        """Test get current goals when no history exists"""
        response = self.client.get("/api/goals/current/nonexistent-session")
        assert response.status_code == 404

    def test_update_thresholds(self):
        """Test update custom thresholds endpoint"""
        custom_thresholds = {"bass_kraft": 0.90, "brillanz": 0.88}

        response = self.client.post("/api/goals/thresholds/test-session", json=custom_thresholds)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify thresholds saved
        assert self.api.custom_thresholds["test-session"]["bass_kraft"] == 0.90


class TestIntegration:
    """Integration tests for complete workflow"""

    async def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        api = MusicalGoalsMonitorAPI()
        api.connection_manager.send_to_session = AsyncMock()

        # 1. Set custom thresholds
        custom = {"bass_kraft": 0.90}
        api.custom_thresholds["production-session"] = custom

        # 2. Process multiple steps
        steps = ["noise_reduction", "eq", "compression", "limiting"]

        for i, step in enumerate(steps):
            goals = {"bass_kraft": 0.85 + i * 0.01, "brillanz": 0.88 + i * 0.01}
            thresholds = {"bass_kraft": 0.85, "brillanz": 0.85}

            await api.update_goals(session_id="production-session", step_name=step, goals=goals, thresholds=thresholds)

        # 3. Verify history
        history = api.history["production-session"]
        assert len(history) == 4

        # 4. Verify last step
        last = history[-1]
        assert last.step_name == "limiting"
        assert last.goals["bass_kraft"] >= 0.88

        # 5. Verify custom thresholds applied
        assert last.thresholds["bass_kraft"] == 0.90

    async def test_violation_tracking(self):
        """Test violation detection and tracking"""
        api = MusicalGoalsMonitorAPI()
        api.connection_manager.send_to_session = AsyncMock()

        # Create scenarios with violations
        scenarios = [
            ("step1", {"bass_kraft": 0.87}, ["bass_kraft"]),  # No violation
            ("step2", {"bass_kraft": 0.82}, ["bass_kraft"]),  # Violation
            ("step3", {"bass_kraft": 0.88}, ["bass_kraft"]),  # Recovered
        ]

        violation_count = 0

        for step_name, goals, goal_names in scenarios:
            thresholds = dict.fromkeys(goal_names, 0.85)

            await api.update_goals(session_id="test", step_name=step_name, goals=goals, thresholds=thresholds)

            snapshot = api.history["test"][-1]
            if len(snapshot.violations) > 0:
                violation_count += 1

        # Should have detected 1 violation (step2)
        assert violation_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
