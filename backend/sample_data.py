import random
import time
from datetime import datetime, timedelta
from collections import deque

STORES = [
    {"id": "chn-rapuram", "name": "Smart Mart - R.A. Puram", "city": "Chennai", "area": "R.A. Puram", "size_factor": 1.2},
    {"id": "pondy", "name": "Smart Mart - Pondicherry", "city": "Pondicherry", "area": "MG Road", "size_factor": 1.0},
    {"id": "chidambaram", "name": "Smart Mart - Chidambaram", "city": "Chidambaram", "area": "Main Road", "size_factor": 0.7},
    {"id": "trichy", "name": "Smart Mart - Trichy", "city": "Trichy", "area": "Thillai Nagar", "size_factor": 0.9},
    {"id": "chn-ramapuram", "name": "Smart Mart - Ramapuram", "city": "Chennai", "area": "Ramapuram", "size_factor": 1.1},
]

class SampleDataGenerator:
    def __init__(self, store_id="chn-rapuram", store_name="Smart Mart", size_factor=1.0):
        self.store_id = store_id
        self.store_name = store_name
        self.size_factor = size_factor
        random.seed(hash(store_id) % 10000)

        self.store_layout = {
            "entrance": {"x": 8, "y": 45, "width": 16, "height": 10, "type": "entrance", "label": "ENTRANCE"},
            "grocery": {"x": 8, "y": 15, "width": 30, "height": 28, "type": "aisle", "label": "GROCERY"},
            "electronics": {"x": 8, "y": 58, "width": 30, "height": 28, "type": "aisle", "label": "ELECTRONICS"},
            "clothing": {"x": 60, "y": 15, "width": 32, "height": 28, "type": "aisle", "label": "CLOTHING"},
            "produce": {"x": 60, "y": 58, "width": 32, "height": 28, "type": "aisle", "label": "PRODUCE"},
            "checkout": {"x": 42, "y": 46, "width": 16, "height": 14, "type": "checkout", "label": "CHECKOUT"},
            "promo_fest": {"x": 40, "y": 15, "width": 18, "height": 14, "type": "promo", "label": "FESTIVE PROMO"},
            "promo_weekend": {"x": 40, "y": 72, "width": 18, "height": 14, "type": "promo", "label": "WEEKEND DEAL"},
            "storage": {"x": 44, "y": 64, "width": 6, "height": 6, "type": "storage", "label": "STOCK"},
        }

        self.promos = {
            "promo_fest": {"name": "Diwali Festive Offer", "discount": "20% OFF Electronics", "start": "2026-09-01", "end": "2026-09-30"},
            "promo_weekend": {"name": "Weekend Grocery Deal", "discount": "Buy 2 Get 1", "start": "2026-09-05", "end": "2026-09-07"},
        }

        self.cameras = [
            {"id": "CAM-01", "name": "Entrance", "zone": "entrance", "status": "online", "res": "1080p", "fps": 30},
            {"id": "CAM-02", "name": "Grocery Aisle", "zone": "grocery", "status": "online", "res": "1080p", "fps": 30},
            {"id": "CAM-03", "name": "Electronics", "zone": "electronics", "status": "online", "res": "4K", "fps": 15},
            {"id": "CAM-04", "name": "Clothing", "zone": "clothing", "status": "online", "res": "1080p", "fps": 30},
            {"id": "CAM-05", "name": "Produce", "zone": "produce", "status": "online", "res": "1080p", "fps": 30},
            {"id": "CAM-06", "name": "Checkout", "zone": "checkout", "status": "online", "res": "1080p", "fps": 30},
            {"id": "CAM-07", "name": "Festive Promo", "zone": "promo_fest", "status": "online", "res": "1080p", "fps": 30},
            {"id": "CAM-08", "name": "Weekend Deal", "zone": "promo_weekend", "status": "online", "res": "1080p", "fps": 30},
        ]

        self.products = [
            {"id": 1, "name": "Basmati Rice 5kg", "category": "Grocery", "shelf": "A1", "max_stock": 50, "price": 450, "barcode": "8901001001101"},
            {"id": 2, "name": "Sunflower Oil 1L", "category": "Grocery", "shelf": "A2", "max_stock": 40, "price": 180, "barcode": "8901001001102"},
            {"id": 3, "name": "Toned Milk 500ml", "category": "Dairy", "shelf": "A3", "max_stock": 60, "price": 25, "barcode": "8901001001103"},
            {"id": 4, "name": "Whole Wheat Bread", "category": "Bakery", "shelf": "A4", "max_stock": 30, "price": 40, "barcode": "8901001001104"},
            {"id": 5, "name": "Smartphone X12", "category": "Electronics", "shelf": "B1", "max_stock": 15, "price": 18999, "barcode": "8901001001105"},
            {"id": 6, "name": "Wireless Earbuds", "category": "Electronics", "shelf": "B2", "max_stock": 25, "price": 1299, "barcode": "8901001001106"},
            {"id": 7, "name": "USB-C Cable", "category": "Electronics", "shelf": "B3", "max_stock": 100, "price": 199, "barcode": "8901001001107"},
            {"id": 8, "name": "Cotton T-Shirt", "category": "Clothing", "shelf": "C1", "max_stock": 80, "price": 499, "barcode": "8901001001108"},
            {"id": 9, "name": "Denim Jeans", "category": "Clothing", "shelf": "C2", "max_stock": 45, "price": 1499, "barcode": "8901001001109"},
            {"id": 10, "name": "Running Shoes", "category": "Footwear", "shelf": "C3", "max_stock": 35, "price": 2499, "barcode": "8901001001110"},
            {"id": 11, "name": "Fresh Apples 1kg", "category": "Produce", "shelf": "D1", "max_stock": 70, "price": 120, "barcode": "8901001001111"},
            {"id": 12, "name": "Tomatoes 1kg", "category": "Produce", "shelf": "D2", "max_stock": 65, "price": 35, "barcode": "8901001001112"},
        ]
        self.next_product_id = 13
        self.max_counters = 6
        self.people = []
        self.footfall_log = deque(maxlen=300)
        self.sales_log = deque(maxlen=300)
        self.live_detections = {}  # camera_id -> {persons, behaviors, timestamp}
        self._init_people()
        self._init_state()

    # ---------- init ----------
    def _init_people(self):
        n = int(random.randint(18, 36) * self.size_factor)
        self.people = []
        for i in range(n):
            self.people.append({
                "id": i, "zone": random.choice(list(self.store_layout.keys())),
                "x": random.randint(5, 95), "y": random.randint(5, 95),
                "moving": random.random() > 0.3,
                "behavior": random.choice(["Standing", "Walking", "Walking", "Standing", "Picking"]),
                "speed": random.uniform(0.2, 0.8),
                "velocity": [random.uniform(-1, 1), random.uniform(-1, 1)]
            })

    def _init_state(self):
        self.current_footfall = len(self.people)
        self.total_visitors_today = int(random.randint(300, 600) * self.size_factor)
        self.total_buyers_today = int(self.total_visitors_today * random.uniform(0.3, 0.55))
        self.hourly_footfall = [int(random.randint(5, 28) * self.size_factor) for _ in range(24)]
        self.history = {"footfall": [], "dwell": [], "queue_total": [], "sales_rate": []}
        self.zone_occupancy = {z: max(0, int(random.randint(2, 14) * self.size_factor)) for z in self.store_layout.keys()}
        self.dwell_times = {z: round(random.uniform(1.5, 9.0), 1) for z in self.store_layout.keys()}
        self.zone_visits_today = {z: int(random.randint(40, 200) * self.size_factor) for z in self.store_layout.keys()}
        self.heatmap = [[random.randint(0, 10) for _ in range(20)] for _ in range(16)]
        self.promo_stats = {
            "promo_fest": {"footfall": int(120 * self.size_factor), "dwell": round(random.uniform(3, 7), 1),
                           "conversions": int(40 * self.size_factor), "revenue": int(45000 * self.size_factor)},
            "promo_weekend": {"footfall": int(150 * self.size_factor), "dwell": round(random.uniform(2, 6), 1),
                              "conversions": int(65 * self.size_factor), "revenue": int(28000 * self.size_factor)},
        }
        self.inventory = {}
        for p in self.products:
            # force a couple out-of-stock for demo
            if p["id"] in (3, 9):
                cur = 0
            else:
                cur = random.randint(int(p["max_stock"] * 0.1), p["max_stock"])
            self.inventory[p["id"]] = {
                "current_stock": cur, "max_stock": p["max_stock"], "price": p["price"],
                "restock_threshold": int(p["max_stock"] * 0.25), "status": "IN_STOCK",
                "last_restock": datetime.now().isoformat()
            }
            self._update_stock_status(p["id"])
        # counters: 6 physical, start with 3 open
        self.queues = {}
        for i in range(1, self.max_counters + 1):
            is_open = i <= 3
            self.queues[i] = {
                "queue_length": random.randint(0, 6) if is_open else 0,
                "avg_service_time_sec": round(random.uniform(45, 90), 0),  # per customer billing
                "avg_wait_time_min": 0.0,
                "status": "open" if is_open else "closed",
                "served_today": int(random.randint(20, 80) * self.size_factor) if is_open else 0,
                "position": {"x": 42 + (i - 1) * 2.5, "y": 50}
            }
        self._recalc_waits()
        self.daily_stats = {
            "total_sales": int(random.randint(60000, 160000) * self.size_factor),
            "total_items_sold": int(random.randint(250, 700) * self.size_factor),
            "avg_dwell_time": round(random.uniform(12, 24), 1),
            "peak_hour": random.randint(11, 20),
            "sales_per_min": round(random.uniform(2, 8) * self.size_factor, 1),
        }
        self.alerts = []
        # seed footfall log
        now = datetime.now()
        for i in range(40):
            ts = now - timedelta(minutes=random.randint(1, 300))
            self.footfall_log.append({
                "date": ts.strftime("%Y-%m-%d"), "time": ts.strftime("%H:%M:%S"),
                "timestamp": ts.isoformat(), "store_id": self.store_id,
                "place": random.choice(list(self.store_layout.keys())),
                "event": random.choice(["entry", "entry", "exit"]),
                "count": random.randint(1, 4)
            })
        self.generate_alerts()

    # ---------- helpers ----------
    def _update_stock_status(self, pid):
        s = self.inventory[pid]
        if s["current_stock"] <= 0:
            s["status"] = "OUT_OF_STOCK"
        elif s["current_stock"] <= s["restock_threshold"]:
            s["status"] = "LOW"
        else:
            s["status"] = "IN_STOCK"

    def _recalc_waits(self):
        for cid, q in self.queues.items():
            if q["status"] != "open":
                q["avg_wait_time_min"] = 0.0
                continue
            # accurate: queue * service_time + randomness (service includes scanning+billing)
            wait_sec = q["queue_length"] * q["avg_service_time_sec"] + random.uniform(0, 20)
            q["avg_wait_time_min"] = round(wait_sec / 60, 1)

    def conversion_rate(self):
        if self.total_visitors_today == 0:
            return 0.0
        return round(self.total_buyers_today / self.total_visitors_today, 3)

    def generate_alerts(self):
        self.alerts = []
        for pid, inv in self.inventory.items():
            prod = next(p for p in self.products if p["id"] == pid)
            if inv["status"] == "OUT_OF_STOCK":
                self.alerts.append({"type": "inventory", "severity": "critical",
                    "message": f"OUT OF STOCK: {prod['name']} (Shelf {prod['shelf']}) - refill immediately",
                    "timestamp": datetime.now().isoformat()})
            elif inv["status"] == "LOW":
                self.alerts.append({"type": "inventory", "severity": "warning",
                    "message": f"Low stock: {prod['name']} ({inv['current_stock']} left, Shelf {prod['shelf']})",
                    "timestamp": datetime.now().isoformat()})
        for cid, q in self.queues.items():
            if q["status"] == "open" and q["queue_length"] >= 6:
                self.alerts.append({"type": "queue", "severity": "warning",
                    "message": f"Counter {cid}: {q['queue_length']} waiting (~{q['avg_wait_time_min']} min) - open another counter",
                    "timestamp": datetime.now().isoformat()})
        for cam in self.cameras:
            if cam["status"] == "offline":
                self.alerts.append({"type": "camera", "severity": "critical",
                    "message": f"Camera {cam['id']} ({cam['name']}) offline",
                    "timestamp": datetime.now().isoformat()})

    # ---------- live update ----------
    def update(self):
        # move people
        for p in self.people:
            if p["moving"]:
                p["x"] = max(0, min(100, p["x"] + p["velocity"][0] * p["speed"]))
                p["y"] = max(0, min(100, p["y"] + p["velocity"][1] * p["speed"]))
                if random.random() < 0.05:
                    p["behavior"] = random.choice(["Standing", "Walking", "Walking", "Picking"])
        # random entries / exits with full date-time-place log
        if random.random() < 0.7:
            n = random.randint(1, 3)
            zone = random.choice(["entrance", "entrance", "grocery", "clothing", "promo_fest", "promo_weekend"])
            now = datetime.now()
            self.total_visitors_today += n
            self.current_footfall = max(0, self.current_footfall + random.randint(-2, 3))
            self.zone_occupancy[zone] = max(0, self.zone_occupancy.get(zone, 0) + random.randint(0, 2))
            self.zone_visits_today[zone] = self.zone_visits_today.get(zone, 0) + n
            self.footfall_log.append({"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                "timestamp": now.isoformat(), "store_id": self.store_id, "place": zone,
                "event": "entry", "count": n})
            # some become buyers
            if random.random() < 0.45:
                b = random.randint(1, n)
                self.total_buyers_today += b
                amt = b * random.randint(200, 1500)
                self.daily_stats["total_sales"] += amt
                self.daily_stats["total_items_sold"] += b * random.randint(1, 4)
                self.sales_log.append({"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                    "timestamp": now.isoformat(), "store_id": self.store_id,
                    "counter_id": random.randint(1, self.max_counters),
                    "items": b, "amount": amt})
        hour = datetime.now().hour
        self.hourly_footfall[hour] = min(120, self.hourly_footfall[hour] + random.randint(0, 2))
        # dwell drift
        for z in self.dwell_times:
            self.dwell_times[z] = round(max(0.5, min(15, self.dwell_times[z] + random.uniform(-0.4, 0.4))), 1)
        # promo drift
        for pk in self.promo_stats:
            ps = self.promo_stats[pk]
            ps["footfall"] += random.randint(0, 2)
            ps["dwell"] = round(max(1, min(12, ps["dwell"] + random.uniform(-0.2, 0.2))), 1)
            if random.random() < 0.3:
                c = random.randint(0, 2)
                ps["conversions"] += c
                ps["revenue"] += c * random.randint(300, 1200)
        # inventory depletion
        for pid in list(self.inventory.keys()):
            if random.random() < 0.06:
                self.inventory[pid]["current_stock"] = max(0, self.inventory[pid]["current_stock"] - random.randint(0, 2))
                self._update_stock_status(pid)
        # queues
        for cid, q in self.queues.items():
            if q["status"] == "open":
                q["queue_length"] = max(0, min(14, q["queue_length"] + random.randint(-2, 2)))
                q["avg_service_time_sec"] = max(30, min(120, q["avg_service_time_sec"] + random.uniform(-3, 3)))
                if random.random() < 0.4:
                    served = random.randint(0, 2)
                    q["served_today"] += served
            else:
                q["queue_length"] = 0
        self._recalc_waits()
        self._auto_adjust_counters()
        # history
        self.history["footfall"].append(self.current_footfall)
        self.history["dwell"].append(self.daily_stats["avg_dwell_time"])
        self.history["queue_total"].append(sum(q["queue_length"] for q in self.queues.values() if q["status"] == "open"))
        self.history["sales_rate"].append(self.daily_stats["sales_per_min"])
        for k in self.history:
            if len(self.history[k]) > 40:
                self.history[k] = self.history[k][-40:]
        self.daily_stats["sales_per_min"] = round(max(0.5, min(20, self.daily_stats["sales_per_min"] + random.uniform(-0.5, 0.5))), 1)
        self.generate_alerts()

    def _auto_adjust_counters(self):
        """Open extra counters when sales/queue high, close when low."""
        open_ids = [c for c, q in self.queues.items() if q["status"] == "open"]
        total_q = sum(self.queues[c]["queue_length"] for c in open_ids)
        avg_q = (total_q / max(1, len(open_ids)))
        rate = self.daily_stats["sales_per_min"]
        self.last_counter_action = "No change - counters optimal"
        # high load: avg queue >=5 or sales rate high
        if (avg_q >= 5 or rate >= 9) and len(open_ids) < self.max_counters:
            for cid, q in self.queues.items():
                if q["status"] == "closed":
                    q["status"] = "open"
                    q["queue_length"] = 1
                    self._recalc_waits()
                    self.last_counter_action = f"Auto-opened Counter {cid} (high sales {rate}/min, avg queue {avg_q:.1f})"
                    break
        elif (avg_q <= 1.2 and rate <= 3 and len(open_ids) > 1):
            # close the emptiest
            emptiest = min(open_ids, key=lambda c: self.queues[c]["queue_length"])
            if self.queues[emptiest]["queue_length"] <= 1:
                self.queues[emptiest]["status"] = "closed"
                self.queues[emptiest]["queue_length"] = 0
                self._recalc_waits()
                self.last_counter_action = f"Auto-closed Counter {emptiest} (low sales {rate}/min, avg queue {avg_q:.1f})"

    # ---------- detection ingest (link YOLO detector) ----------
    def ingest_detection(self, payload: dict):
        """payload: {camera_id, timestamp, persons:[{track_id, behavior, bbox}], zone?}"""
        cam_id = payload.get("camera_id", "CAM-01")
        persons = payload.get("persons", [])
        ts = payload.get("timestamp", datetime.now().isoformat())
        self.live_detections[cam_id] = {"timestamp": ts, "persons": persons, "count": len(persons)}
        # drive zone occupancy from real detections when available
        cam = next((c for c in self.cameras if c["id"] == cam_id), None)
        if cam:
            zone = cam["zone"]
            self.zone_occupancy[zone] = len(persons)
            # update behaviors of simulated people sample
            for i, pr in enumerate(persons[:5]):
                if self.people:
                    self.people[(pr.get("track_id", i)) % len(self.people)]["behavior"] = pr.get("behavior", "Standing")
        return {"ok": True, "store_id": self.store_id, "camera_id": cam_id, "count": len(persons)}

    # ---------- getters ----------
    def get_shopper_analytics(self):
        return {
            "store_id": self.store_id, "store_name": self.store_name,
            "current_footfall": self.current_footfall,
            "total_visitors_today": self.total_visitors_today,
            "total_buyers_today": self.total_buyers_today,
            "conversion_rate": self.conversion_rate(),
            "hourly_footfall": self.hourly_footfall,
            "zone_occupancy": self.zone_occupancy,
            "zone_visits_today": self.zone_visits_today,
            "dwell_times": self.dwell_times,
            "heatmap": self.heatmap,
            "daily_stats": self.daily_stats,
            "history": self.history,
            "people": self.people,
            "layout": self.store_layout,
            "footfall_log": list(self.footfall_log)[-30:],
            "live_detections": self.live_detections,
        }

    def get_promotions(self):
        out = []
        for pid, meta in self.promos.items():
            st = self.promo_stats[pid]
            conv = round(st["conversions"] / max(1, st["footfall"]), 3)
            out.append({"id": pid, "name": meta["name"], "discount": meta["discount"],
                        "start": meta["start"], "end": meta["end"],
                        "footfall": st["footfall"], "dwell_min": st["dwell"],
                        "conversions": st["conversions"], "conversion_rate": conv,
                        "revenue": st["revenue"],
                        "layout": self.store_layout.get(pid, {})})
        out.sort(key=lambda x: x["revenue"], reverse=True)
        if out:
            out[0]["best"] = True
        return {"promotions": out}

    def get_inventory_status(self):
        items = []
        for p in self.products:
            inv = self.inventory[p["id"]]
            status = inv["status"]
            items.append({**p, "current_stock": inv["current_stock"],
                          "stock_status": status,
                          "stock_percentage": round(inv["current_stock"] / inv["max_stock"] * 100) if inv["max_stock"] else 0,
                          "restock_threshold": inv["restock_threshold"],
                          "last_restock": inv["last_restock"]})
        return {"items": items, "total_products": len(self.products),
                "out_of_stock_count": sum(1 for i in self.inventory.values() if i["status"] == "OUT_OF_STOCK"),
                "low_count": sum(1 for i in self.inventory.values() if i["status"] == "LOW"),
                "in_stock_count": sum(1 for i in self.inventory.values() if i["status"] == "IN_STOCK"),
                "total_inventory_value": sum(self.inventory[p["id"]]["current_stock"] * p["price"] for p in self.products)}

    def get_queue_status(self):
        counters = []
        for cid, q in self.queues.items():
            billing_min = round(q["avg_service_time_sec"] / 60, 1)
            counters.append({"counter_id": cid, "queue_length": q["queue_length"],
                             "avg_wait_time_min": q["avg_wait_time_min"],
                             "avg_billing_time_min": billing_min,
                             "avg_service_time_sec": round(q["avg_service_time_sec"], 0),
                             "status": q["status"], "served_today": q["served_today"],
                             "position": q["position"]})
        open_q = [q for q in self.queues.values() if q["status"] == "open"]
        total_waiting = sum(q["queue_length"] for q in open_q)
        avg_wait = round(sum(q["avg_wait_time_min"] for q in open_q) / max(1, len(open_q)), 1)
        avg_bill = round(sum(q["avg_service_time_sec"] for q in open_q) / max(1, len(open_q)) / 60, 1)
        return {"counters": counters, "total_waiting": total_waiting,
                "average_wait_time_min": avg_wait, "average_billing_time_min": avg_bill,
                "open_counters": len(open_q), "max_counters": self.max_counters,
                "sales_per_min": self.daily_stats["sales_per_min"],
                "auto_action": getattr(self, "last_counter_action", "Monitoring..."),
                "recommendation": getattr(self, "last_counter_action", "Monitoring...")}

    def get_conversion(self):
        return {"store_id": self.store_id, "store_name": self.store_name,
                "visitors": self.total_visitors_today, "buyers": self.total_buyers_today,
                "conversion_rate": self.conversion_rate(),
                "conversion_pct": round(self.conversion_rate() * 100, 1),
                "promo_conversions": {k: {"rate": round(v["conversions"] / max(1, v["footfall"]), 3),
                                          **v} for k, v in self.promo_stats.items()}}

    def get_footfall_log(self, limit=50):
        return {"store_id": self.store_id, "logs": list(self.footfall_log)[-limit:][::-1]}

    def get_alerts(self):
        return self.alerts

    def get_store_overview(self):
        online = sum(1 for c in self.cameras if c["status"] == "online")
        return {"store_id": self.store_id, "store_name": self.store_name,
                "timestamp": datetime.now().isoformat(), "total_cameras": len(self.cameras),
                "active_cameras": online, "edge_devices": 3, "system_status": "operational", "uptime": "99.7%"}

    def get_cameras(self):
        return self.cameras

    def get_camera_simulation(self, camera_id):
        cam = next((c for c in self.cameras if c["id"] == camera_id), None)
        if not cam:
            return None
        live = self.live_detections.get(camera_id)
        zone_people = [p for p in self.people if p["zone"] == cam["zone"]]
        return {"camera_id": cam["id"], "zone": cam["zone"], "status": cam["status"], "fps": cam["fps"],
                "people_count": live["count"] if live else len(zone_people),
                "live": bool(live), "live_timestamp": live["timestamp"] if live else None,
                "people": zone_people, "zone_bounds": self.store_layout.get(cam["zone"], {})}

    def get_ai_models(self):
        return [
            {"name": "YOLOv8-nano (People Detection)", "accuracy": 96.2, "fps": 30, "status": "active",
             "framework": "PyTorch / ONNX", "description": "Real-time person detection and counting at entry/exit",
             "target": "Shopper Analytics"},
            {"name": "DeepSORT (Multi-Object Tracking)", "accuracy": 91.5, "fps": 25, "status": "active",
             "framework": "TensorFlow", "description": "Tracks shopper paths and dwell time across zones",
             "target": "Shopper Analytics"},
            {"name": "Shelf-OOS Detector (ResNet18)", "accuracy": 88.9, "fps": 15, "status": "active",
             "framework": "TensorFlow Lite", "description": "Detects OUT-OF-STOCK and low-stock from shelf cameras",
             "target": "Inventory"},
            {"name": "Queue Length CNN (EfficientNet)", "accuracy": 90.3, "fps": 20, "status": "active",
             "framework": "ONNX Runtime", "description": "Estimates queue length, wait + billing time from overhead cams",
             "target": "Queue Management"},
            {"name": "Hand-Gesture Picking Detector", "accuracy": 87.4, "fps": 22, "status": "active",
             "framework": "PyTorch", "description": "Open/close hand in front of camera = picking event",
             "target": "Promo Zones"},
            {"name": "Anomaly Detection (LSTM)", "accuracy": 93.1, "fps": 60, "status": "active",
             "framework": "TensorFlow", "description": "Unusual crowding alerts", "target": "Security"},
        ]

    def get_sales(self):
        return list(self.sales_log)[-50:][::-1]

    # CRUD
    def add_product(self, data):
        product = {"id": self.next_product_id, "name": data["name"],
                   "category": data.get("category", "General"), "shelf": data.get("shelf", "E1"),
                   "max_stock": int(data.get("max_stock", 50)), "price": float(data.get("price", 100)),
                   "barcode": data.get("barcode") or f"890100100111{self.next_product_id}"}
        self.products.append(product)
        self.inventory[product["id"]] = {"current_stock": int(data.get("current_stock", product["max_stock"])),
            "max_stock": product["max_stock"], "price": product["price"],
            "restock_threshold": int(product["max_stock"] * 0.25), "status": "IN_STOCK",
            "last_restock": datetime.now().isoformat()}
        self._update_stock_status(product["id"])
        self.next_product_id += 1
        self.generate_alerts()
        return product

    def update_product(self, pid, data):
        prod = next((p for p in self.products if p["id"] == pid), None)
        if not prod:
            return None
        for k in ("name", "category", "shelf", "barcode"):
            if k in data and data[k] is not None:
                prod[k] = data[k]
        if data.get("max_stock") is not None:
            prod["max_stock"] = int(data["max_stock"])
        if data.get("price") is not None:
            prod["price"] = float(data["price"])
            self.inventory[pid]["price"] = prod["price"]
        if data.get("current_stock") is not None:
            self.inventory[pid]["current_stock"] = int(data["current_stock"])
        self.inventory[pid]["max_stock"] = prod["max_stock"]
        self.inventory[pid]["restock_threshold"] = int(prod["max_stock"] * 0.25)
        self._update_stock_status(pid)
        self.generate_alerts()
        return prod

    def delete_product(self, pid):
        if not any(p["id"] == pid for p in self.products):
            return False
        self.products = [p for p in self.products if p["id"] != pid]
        self.inventory.pop(pid, None)
        self.generate_alerts()
        return True

    def restock(self, pid, qty=None):
        if pid not in self.inventory:
            return {"status": "error", "message": "Product not found"}
        prod = next(p for p in self.products if p["id"] == pid)
        if qty is None:
            qty = self.inventory[pid]["max_stock"] - self.inventory[pid]["current_stock"]
        self.inventory[pid]["current_stock"] = min(self.inventory[pid]["max_stock"],
            self.inventory[pid]["current_stock"] + int(qty))
        self.inventory[pid]["last_restock"] = datetime.now().isoformat()
        self._update_stock_status(pid)
        self.generate_alerts()
        return {"status": "success", "message": f"Restocked {prod['name']} (+{qty})"}

    # reports
    def report_rows(self, rtype):
        now = datetime.now()
        if rtype == "footfall":
            return [{"date": e["date"], "time": e["time"], "store_id": e["store_id"],
                     "place_zone": e["place"], "event": e["event"], "count": e["count"]} for e in self.footfall_log]
        if rtype == "inventory":
            inv = self.get_inventory_status()["items"]
            return [{"id": i["id"], "name": i["name"], "category": i["category"], "shelf": i["shelf"],
                     "price": i["price"], "current_stock": i["current_stock"], "max_stock": i["max_stock"],
                     "status": i["stock_status"], "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")} for i in inv]
        if rtype == "queue":
            q = self.get_queue_status()["counters"]
            return [{"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                     "counter_id": c["counter_id"], "status": c["status"], "queue_length": c["queue_length"],
                     "wait_min": c["avg_wait_time_min"], "billing_min": c["avg_billing_time_min"],
                     "served_today": c["served_today"]} for c in q]
        if rtype == "promo":
            return [{"promo_id": p["id"], "name": p["name"], "discount": p["discount"],
                     "footfall": p["footfall"], "dwell_min": p["dwell_min"], "conversions": p["conversions"],
                     "conversion_rate": p["conversion_rate"], "revenue": p["revenue"]} for p in self.get_promotions()["promotions"]]
        if rtype == "sales":
            return list(self.sales_log)
        if rtype == "conversion":
            c = self.get_conversion()
            return [{"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                     "store_id": c["store_id"], "visitors": c["visitors"], "buyers": c["buyers"],
                     "conversion_rate": c["conversion_rate"]}]
        return []
