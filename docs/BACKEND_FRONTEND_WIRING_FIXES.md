# Backend-Frontend Wiring - Erforderliche Fixes

**Datum:** 13. Februar 2026  
**Status:** 🚨 Kritische Probleme gefunden

---

## 🚨 Kritische Probleme

### Problem 1: User Feedback Endpoint Mismatch

**Symptom:**
Frontend kann User-Feedback nicht senden (404 Error erwartet)

**Aktueller Zustand:**
```javascript
// frontend/api.js
export async function sendUserFeedback(feedback) {
  const res = await fetch(`${API_URL}/user_feedback`, {  // ❌ FALSCH
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(feedback)
  });
  return res.json();
}
```

```python
# backend/api.py
@app.post("/feedback")  # ✅ Existiert als /feedback
async def submit_feedback(request: Request):
    # ... implementiert
```

**Fix Option A:** Frontend anpassen (empfohlen)
```javascript
// frontend/api.js
export async function sendUserFeedback(feedback) {
  const res = await fetch(`${API_URL}/feedback`, {  // ✅ KORRIGIERT
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(feedback)
  });
  return res.json();
}
```

**Fix Option B:** Backend Alias hinzufügen
```python
# backend/api.py
@app.post("/user_feedback")  # Alias für Kompatibilität
async def submit_user_feedback(request: Request):
    return await submit_feedback(request)
```

---

### Problem 2: Batch API - Port & Framework Mismatch

**Symptom:**
- Frontend Batch-Workflow kann nicht gestartet werden
- Connection refused auf Port 8000 für Batch-Endpunkte

**Aktueller Zustand:**

**Frontend erwartet (Port 8000, FastAPI):**
```javascript
// frontend/BatchWorkflowPanel.jsx
POST http://localhost:8000/batch/start
GET  http://localhost:8000/batch/status/{batchId}
GET  http://localhost:8000/batch/audit/{batchId}
```

**Backend bietet (Port 5000, Flask):**
```python
# backend/batch_api.py (Flask auf Port 5000!)
@app.route("/batch/start", methods=["POST"])
@app.route("/batch/status", methods=["GET"])  # OHNE {batchId}!
@app.route("/batch/audit", methods=["GET"])   # OHNE {batchId}!
```

**Problem-Details:**
1. **Zwei separate Server:** FastAPI (8000) + Flask (5000)
2. **Keine Batch-IDs:** Flask-API hat single-batch Design
3. **API-Design Mismatch:** Frontend erwartet Multi-Batch-Support

**Fix Option A: Batch-API in FastAPI integrieren (empfohlen)**

1. Erstelle `backend/api/rest/batch_endpoints.py`:
```python
from fastapi import APIRouter, HTTPException
import uuid
from typing import Dict
import threading

router = APIRouter(prefix="/batch", tags=["batch"])

# Batch-Status-Verwaltung (Multi-Batch-Support)
batch_jobs: Dict[str, dict] = {}

@router.post("/start")
async def start_batch():
    """Startet einen neuen Batch-Job"""
    batch_id = str(uuid.uuid4())
    batch_jobs[batch_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "last_file": None
    }
    # Starte Batch-Worker in Thread
    threading.Thread(
        target=batch_worker, 
        args=(batch_id,), 
        daemon=True
    ).start()
    return {"batch_id": batch_id, "status": "started"}

@router.get("/status/{batch_id}")
async def batch_status(batch_id: str):
    """Gibt Status eines Batch-Jobs zurück"""
    if batch_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch-ID nicht gefunden")
    return batch_jobs[batch_id]

@router.get("/audit/{batch_id}")
async def batch_audit(batch_id: str):
    """Gibt Audit-Report für Batch-Job zurück"""
    if batch_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch-ID nicht gefunden")
    # Implementierung für Audit-Report
    audits = []  # TODO: Audit-Files sammeln
    return {"batch_id": batch_id, "audits": audits}

def batch_worker(batch_id: str):
    """Worker-Thread für Batch-Verarbeitung"""
    # TODO: Implementierung aus batch_api.py übernehmen
    pass
```

2. Integriere in `backend/api.py`:
```python
from backend.api.rest.batch_endpoints import router as batch_router

app.include_router(batch_router)  # Registriere Batch-Endpunkte
```

**Fix Option B: Frontend auf Port 5000 umleiten (nicht empfohlen)**
```javascript
// frontend/BatchWorkflowPanel.jsx
const BATCH_API_URL = 'http://localhost:5000';  // Separate Flask-API

// Ändere alle Batch-Calls:
fetch(`${BATCH_API_URL}/batch/start`, ...)
fetch(`${BATCH_API_URL}/batch/status`, ...)  // Ohne {batchId}
fetch(`${BATCH_API_URL}/batch/audit`, ...)   // Ohne {batchId}
```

---

### Problem 3: Fehlende Frontend-Integration für Backend-Endpunkte

**Backend-Endpunkte OHNE Frontend-Anbindung:**

```python
# backend/api.py - Nicht genutzte Endpunkte:
GET /health                    # Health Check
GET /stream                    # Audio Streaming
GET /download_report           # SOTA Report Download
GET /download_quality_log      # Quality Log JSON
GET /download_quality_log_csv  # Quality Log CSV
GET /download_log              # Backend Log Download
```

**Empfohlene Frontend-Erweiterungen:**

```javascript
// frontend/api.js - Hinzufügen:

// Health Check (für Status-Monitoring)
export async function healthCheck() {
  const res = await fetch(`${API_URL}/health`);
  return res.json();
}

// Download-Funktionen
export async function downloadReport() {
  window.open(`${API_URL}/download_report`, '_blank');
}

export async function downloadQualityLog(format = 'json') {
  const endpoint = format === 'csv' 
    ? '/download_quality_log_csv' 
    : '/download_quality_log';
  window.open(`${API_URL}${endpoint}`, '_blank');
}

export async function downloadBackendLog() {
  window.open(`${API_URL}/download_log`, '_blank');
}

// Audio Streaming (für Preview/Playback)
export async function streamAudio(audioId) {
  return `${API_URL}/stream?id=${audioId}`;
}
```

---

## ✅ Korrekt verdrahtete Endpunkte

Folgende Endpunkte funktionieren bereits:

| Frontend Funktion | Backend Endpunkt | Methode | Status |
|-------------------|------------------|---------|--------|
| `importAudio()` | `/import` | POST | ✅ OK |
| `analyzeAudio()` | `/analyze` | POST | ✅ OK |
| `processAudio()` | `/process` | POST | ✅ OK |
| `exportAudio()` | `/export` | POST | ✅ OK |
| `magicButtonAudio()` | `/magic_button` | POST | ✅ OK |
| `systemCheck()` | `/systemcheck` | GET | ✅ OK |

---

## 📋 Implementierungs-Checkliste

### Schnelle Fixes (< 30 Min)

- [ ] **Fix 1A:** Ändere Frontend `/user_feedback` → `/feedback` ([frontend/api.js](../frontend/api.js#L12))
- [ ] **Fix 3:** Füge Health-Check & Download-Funktionen zu frontend/api.js hinzu

### Mittelfristige Fixes (2-4 Stunden)

- [ ] **Fix 2A:** Migriere Batch-API von Flask nach FastAPI
  1. Erstelle `backend/api/rest/batch_endpoints.py`
  2. Implementiere Multi-Batch-Support mit UUIDs
  3. Registriere Router in `backend/api.py`
  4. Teste mit Frontend BatchWorkflowPanel.jsx

- [ ] **Testing:** Schreibe Integration-Tests für alle Endpunkte
  ```python
  # tests/integration/test_api_endpoints.py
  def test_feedback_endpoint():
      response = client.post("/feedback", json={"rating": 5, "text": "Great!"})
      assert response.status_code == 200
  
  def test_batch_workflow():
      # Start Batch
      response = client.post("/batch/start")
      batch_id = response.json()["batch_id"]
      
      # Check Status
      response = client.get(f"/batch/status/{batch_id}")
      assert response.status_code == 200
  ```

---

## 🚀 Migration Plan: Flask → FastAPI (Batch API)

### Phase 1: Parallel-Betrieb (Woche 1)
1. Implementiere FastAPI Batch-Endpunkte (Port 8000)
2. Behalte Flask-API (Port 5000) als Fallback
3. Frontend nutzt FastAPI (Port 8000)

### Phase 2: Testing (Woche 2)
1. E2E-Tests für alle Batch-Workflows
2. Load-Testing (multiple concurrent batches)
3. Error-Handling validieren

### Phase 3: Deprecation (Woche 3)
1. Flask batch_api.py als `@deprecated` markieren
2. Dokumentation aktualisieren
3. Nach 2 Wochen: Flask-API entfernen

---

## 📝 Dokumentations-Updates

Nach Implementation der Fixes:

1. **API-Dokumentation aktualisieren:**
   - [docs/api/PYTHON_API.md](../api/PYTHON_API.md)
   - Füge REST-API-Referenz hinzu (GET/POST Endpunkte)

2. **Troubleshooting Guide ergänzen:**
   - [docs/guides/TROUBLESHOOTING.md](../guides/TROUBLESHOOTING.md)
   - Problem: "404 bei /user_feedback" → Lösung hinzufügen

3. **Architecture Docs:**
   - [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md)
   - Aktualisiere API-Layer Diagramm

---

## 💡 Best Practices für zukünftige API-Entwicklung

1. **Single API Framework:** Verwende nur FastAPI (NICHT Flask + FastAPI)
2. **API-First Design:** Definiere OpenAPI Schema VOR Implementation
3. **Automatische Tests:** Jeder Endpunkt braucht Integration-Test
4. **API-Versionierung:** `/api/v1/endpoint` für Breaking Changes
5. **Type Safety:** Pydantic Models für alle Request/Response Bodies

---

**© 2026 Aurik Audio Restoration System**  
**Backend-Frontend Wiring Analysis & Fixes**
