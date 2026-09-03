from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import csv
import io
from datetime import datetime
from .sample_data import SampleDataGenerator, STORES

app = FastAPI(title="Retail Intelligence Platform - SIH26179 Centralised")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GENERATORS = {}
for s in STORES:
    GENERATORS[s["id"]] = SampleDataGenerator(store_id=s["id"], store_name=s["name"], size_factor=s["size_factor"])

DEFAULT_STORE = STORES[0]["id"]
connected = set()  # (websocket, store_id)

def G(store_id: str = None) -> SampleDataGenerator:
    return GENERATORS.get(store_id or DEFAULT_STORE, GENERATORS[DEFAULT_STORE])

class ProductCreate(BaseModel):
    name: str
    category: str = "General"
    shelf: str = "E1"
    max_stock: int = 50
    price: float = 100
    current_stock: int = 0
    barcode: str = None

class ProductUpdate(BaseModel):
    name: str = None
    category: str = None
    shelf: str = None
    max_stock: int = None
    price: float = None
    current_stock: int = None
    barcode: str = None

class DetectionIngest(BaseModel):
    store_id: str = DEFAULT_STORE
    camera_id: str = "CAM-01"
    timestamp: str = None
    persons: list = []

@app.on_event("startup")
async def startup():
    asyncio.create_task(periodic_update())

async def periodic_update():
    while True:
        for gen in GENERATORS.values():
            gen.update()
        dead = set()
        for ws, sid in list(connected):
            try:
                await ws.send_text(json.dumps(payload_for(sid, "update")))
            except Exception:
                dead.add((ws, sid))
        connected.difference_update(dead)
        await asyncio.sleep(3)

def payload_for(store_id, msg_type):
    g = G(store_id)
    return {
        "type": msg_type, "store_id": store_id,
        "shopper": g.get_shopper_analytics(),
        "promotions": g.get_promotions(),
        "inventory": g.get_inventory_status(),
        "queue": g.get_queue_status(),
        "conversion": g.get_conversion(),
        "footfall_log": g.get_footfall_log(30),
        "alerts": g.get_alerts(),
        "overview": g.get_store_overview(),
        "cameras": g.get_cameras(),
        "ai_models": g.get_ai_models(),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws")
async def ws_default(websocket: WebSocket):
    await websocket.accept()
    params = dict(websocket.query_params)
    sid = params.get("store_id", DEFAULT_STORE)
    connected.add((websocket, sid))
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                d = json.loads(msg)
                if d.get("store_id"):
                    connected.discard((websocket, sid))
                    sid = d["store_id"]
                    connected.add((websocket, sid))
            except Exception:
                pass
            await websocket.send_text(json.dumps(payload_for(sid, "initial")))
    except WebSocketDisconnect:
        connected.discard((websocket, sid))

@app.get("/api/stores")
async def list_stores():
    out = []
    for s in STORES:
        g = G(s["id"])
        out.append({**s, "visitors_today": g.total_visitors_today,
                    "conversion_rate": g.conversion_rate(),
                    "open_counters": sum(1 for q in g.queues.values() if q["status"] == "open"),
                    "out_of_stock": sum(1 for i in g.inventory.values() if i["status"] == "OUT_OF_STOCK")})
    return {"stores": out, "default": DEFAULT_STORE}

def sid(q: str = Query(default=None, alias="store_id")):
    return q or DEFAULT_STORE

@app.get("/api/shopper-analytics")
async def shopper(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_shopper_analytics()

@app.get("/api/promotions")
async def promos(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_promotions()

@app.get("/api/conversion")
async def conversion(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_conversion()

@app.get("/api/footfall-log")
async def footfall(store_id: str = Query(default=DEFAULT_STORE), limit: int = 50):
    return G(store_id).get_footfall_log(limit)

@app.get("/api/inventory")
async def inventory(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_inventory_status()

@app.post("/api/inventory")
async def create_product(product: ProductCreate, store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).add_product(product.dict())

@app.put("/api/inventory/{pid}")
async def update_product(pid: int, product: ProductUpdate, store_id: str = Query(default=DEFAULT_STORE)):
    u = G(store_id).update_product(pid, product.dict(exclude_none=True))
    if not u:
        raise HTTPException(404, "Product not found")
    return u

@app.delete("/api/inventory/{pid}")
async def delete_product(pid: int, store_id: str = Query(default=DEFAULT_STORE)):
    if not G(store_id).delete_product(pid):
        raise HTTPException(404, "Product not found")
    return {"status": "success"}

@app.get("/api/queue-status")
async def queue_status(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_queue_status()

@app.get("/api/alerts")
async def alerts(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_alerts()

@app.get("/api/overview")
async def overview(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_store_overview()

@app.get("/api/cameras")
async def cameras(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_cameras()

@app.get("/api/cameras/{cam_id}/simulate")
async def cam_sim(cam_id: str, store_id: str = Query(default=DEFAULT_STORE)):
    f = G(store_id).get_camera_simulation(cam_id)
    if not f:
        raise HTTPException(404, "Camera not found")
    return f

@app.get("/api/ai-models")
async def ai_models():
    return G(DEFAULT_STORE).get_ai_models()

@app.get("/api/sales")
async def sales(store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).get_sales()

@app.get("/api/simulate-restock/{pid}")
async def restock(pid: int, store_id: str = Query(default=DEFAULT_STORE)):
    return G(store_id).restock(pid)

# ---- link YOLO camera detector to dashboard ----
@app.post("/api/detection/ingest")
async def ingest(data: DetectionIngest):
    g = G(data.store_id)
    return g.ingest_detection(data.dict())

@app.get("/api/detection/live")
async def live(store_id: str = Query(default=DEFAULT_STORE)):
    g = G(store_id)
    return {"store_id": store_id, "live": g.live_detections, "timestamp": datetime.now().isoformat()}

# ---- reports: download timeline CSVs ----
@app.get("/api/reports/{rtype}")
async def report(rtype: str, store_id: str = Query(default=DEFAULT_STORE), fmt: str = "json"):
    g = G(store_id)
    rows = g.report_rows(rtype)
    if fmt == "csv":
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            buf.write("no_data\n")
        buf.seek(0)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={store_id}_{rtype}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"})
    return {"store_id": store_id, "type": rtype, "count": len(rows), "rows": rows}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
