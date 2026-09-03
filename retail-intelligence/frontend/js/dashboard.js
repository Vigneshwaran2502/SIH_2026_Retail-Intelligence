// Centralised multi-store dashboard
// Cam feed rotates every 2s, zone thermal refreshes every 3s (synced labels)

let state = {
  data:null, stores:[], storeId:null, ws:null,
  activeTab:'overview', camIndex:0, pinnedCam:null,
  ovCam:null, ovCtx:null, thermal:null, thermalCtx:null,
  mapCanvas:null, mapCtx:null, camMain:null, camMainCtx:null,
  frame:0, invFilter:'all', lastHeat:0,
};
let charts = {};
const $ = id => document.getElementById(id);
const cap = s => s.charAt(0).toUpperCase()+s.slice(1);
const fmtN = n => n>=1000 ? (n/1000).toFixed(1)+'k' : String(n);
const api = (p)=> fetch(p).then(r=>r.json());
const apiStore = (p)=> {
  const sep = p.includes('?')?'&':'?';
  return api(p+sep+'store_id='+state.storeId);
};

document.addEventListener('DOMContentLoaded', async ()=>{
  initTabs(); clock(); setInterval(clock,1000); invEvents();
  await loadStores();
  connectWS();
  setInterval(tick2s, 2000);   // camera rotation
  setInterval(tickAnim, 120);   // smooth canvas anim
  setInterval(()=>{ state.lastHeat = Date.now(); }, 3000); // heatmap heartbeat
});

function clock(){
  $('current-time').textContent = new Date().toLocaleString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit',day:'2-digit',month:'short'});
}

async function loadStores(){
  try{
    const d = await api('/api/stores');
    state.stores = d.stores; state.storeId = d.default;
    const sel = $('store-select');
    sel.innerHTML = d.stores.map(s=>`<option value="${s.id}">${s.name} — ${s.city}</option>`).join('');
    sel.value = state.storeId;
    sel.onchange = ()=>{ state.storeId = sel.value; state.camIndex=0; state.pinnedCam=null; connectWS(); refreshAll(); };
    await refreshAll();
  }catch(e){ console.error(e); }
}

function connectWS(){
  try{ state.ws && state.ws.close(); }catch(e){}
  try{
    state.ws = new WebSocket(`ws://${location.host}/ws?store_id=${state.storeId}`);
    state.ws.onmessage = ev => { state.data = JSON.parse(ev.data); render(state.data); };
    state.ws.onclose = ()=> setTimeout(pollOnce, 2500);
    state.ws.onerror = ()=> setTimeout(pollOnce, 2500);
  }catch(e){ pollOnce(); }
}
async function pollOnce(){ await refreshAll(); setTimeout(pollOnce, 3000); }

async function refreshAll(){
  if(!state.storeId) return;
  try{
    const [shopper,promotions,inventory,queue,conversion,footfall_log,alerts,overview,cameras,ai] = await Promise.all([
      apiStore('/api/shopper-analytics'), apiStore('/api/promotions'), apiStore('/api/inventory'),
      apiStore('/api/queue-status'), apiStore('/api/conversion'), apiStore('/api/footfall-log?limit=30'),
      apiStore('/api/alerts'), apiStore('/api/overview'), apiStore('/api/cameras'), api('/api/ai-models')]);
    state.data = {shopper,promotions,inventory,queue,conversion,footfall_log,alerts,overview,cameras,ai_models:ai};
    render(state.data);
  }catch(e){ console.error(e); }
}

function initTabs(){
  document.querySelectorAll('.tab-btn').forEach(b=>b.onclick=()=>{
    state.activeTab=b.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(x=>x.classList.toggle('active',x===b));
    document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
    $('tab-'+state.activeTab).classList.add('active');
    if(state.data) render(state.data);
  });
}

// ---------- rotation: camera every 2s ----------
function tick2s(){
  if(!state.data||!state.data.cameras) return;
  if(state.pinnedCam) return; // user pinned
  state.camIndex = (state.camIndex+1)%state.data.cameras.length;
  if(state.activeTab==='overview') drawOvCam(true);
  if(state.activeTab==='cameras') drawCamMain(true);
}
function tickAnim(){
  if(!state.data) return;
  state.frame++;
  if(state.activeTab==='overview'){ drawOvCam(false); drawThermal(); }
  if(state.activeTab==='cameras'){ drawCamMain(false); }
  if(state.activeTab==='map'){ drawBigMap(); }
}

function render(d){
  head(d); overview(d);
  if(state.activeTab==='cameras') camerasTab(d);
  if(state.activeTab==='map') drawBigMap();
  if(state.activeTab==='promos') promosTab(d);
  if(state.activeTab==='inventory') invTable();
  if(state.activeTab==='queues') queuesTab(d);
  if(state.activeTab==='models') modelsTab(d);
  if(state.activeTab==='alerts') alertsTab(d);
  const b=$('alert-badge'); b.textContent=d.alerts.length; b.style.display=d.alerts.length?'inline-block':'none';
}

function head(d){
  $('system-status').textContent='Operational';
  $('camera-count').textContent=`${d.overview.active_cameras}/${d.overview.total_cameras}`;
  $('live-count').textContent=d.shopper.current_footfall;
  $('head-conv').textContent=(d.conversion.conversion_rate*100).toFixed(1)+'%';
  const st = state.stores.find(s=>s.id===state.storeId);
  if(st) $('store-banner').innerHTML=`<span>🏬 <b>${st.name}</b></span><span>📍 ${st.area}, ${st.city}</span><span>👥 ${d.shopper.total_visitors_today} visitors</span><span>✅ ${(d.conversion.conversion_rate*100).toFixed(1)}% conversion</span><span>⏱ ${d.queue.average_wait_time_min}m wait • 🧾 ${d.queue.average_billing_time_min}m billing</span>`;
}

// ---------- overview ----------
let ovChart=null;
function overview(d){
  const s=d.shopper,q=d.queue,inv=d.inventory,c=d.conversion;
  $('ov-footfall').textContent=s.current_footfall;
  $('ov-visitors').textContent=s.total_visitors_today;
  $('ov-buyers').textContent=c.buyers;
  $('ov-conversion').textContent=c.conversion_pct+'%';
  $('ov-wait').textContent=q.average_wait_time_min+'m';
  $('ov-bill').textContent=q.average_billing_time_min+'m';
  $('ov-oos').textContent=inv.out_of_stock_count;
  $('ov-sales').textContent='₹'+fmtN(s.daily_stats.total_sales);
  // chart
  const cv=$('overview-chart'); if(cv){
    const h=s.history||{}; const f=(h.footfall||[]).slice(-40), sr=(h.sales_rate||[]).slice(-40);
    const labels=f.map((_,i)=>`${i*3}s`);
    if(ovChart){ ovChart.data.labels=labels; ovChart.data.datasets[0].data=f; ovChart.data.datasets[1].data=sr; ovChart.update('none'); }
    else ovChart=new Chart(cv.getContext('2d'),{type:'line',data:{labels,datasets:[
      {label:'Footfall',data:f,borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.1)',fill:true,tension:.4,pointRadius:0},
      {label:'Sales/min',data:sr,borderColor:'#00ff88',tension:.4,pointRadius:0,yAxisID:'y2'}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8a94a8'}}},
      scales:{x:{ticks:{color:'#8a94a8',maxTicksLimit:8},grid:{color:'rgba(15,35,70,.10)'}},
      y:{ticks:{color:'#8a94a8'},grid:{color:'rgba(15,35,70,.10)'},beginAtZero:true},
      y2:{position:'right',ticks:{color:'#00ff88'},grid:{drawOnChartArea:false},beginAtZero:true}}}});
  }
  // dwell
  const mx=Math.max(...Object.values(s.dwell_times),1);
  $('zone-dwell-list').innerHTML=Object.entries(s.dwell_times).sort((a,b)=>b[1]-a[1]).map(([z,v])=>
    `<div class="list-item"><span class="name">${s.layout[z]?s.layout[z].label:z}</span><div class="progress-bar"><div class="progress-fill blue" style="width:${v/mx*100}%"></div></div><span class="val">${v.toFixed(1)}m</span></div>`).join('');
  // footfall log date-time-place
  $('footfall-body').innerHTML=(d.footfall_log.logs||[]).slice(0,12).map(e=>
    `<tr><td>${e.date}</td><td>${e.time}</td><td>${s.layout[e.place]?s.layout[e.place].label:e.place}</td><td>${e.event}</td><td>${e.count}</td></tr>`).join('');
  // promo mini
  $('ov-promo-list').innerHTML=d.promotions.promotions.map(p=>
    `<div class="list-item"><span class="name">${p.best?'⭐ ':''}${p.name}</span><span class="val">${(p.conversion_rate*100).toFixed(1)}%</span><span class="val" style="color:#00ff88">₹${fmtN(p.revenue)}</span></div>`).join('');
  ensureOvCam(); ensureThermal();
}

// ---------- overview camera (2s rotation) ----------
function curCam(){
  const cams=state.data.cameras; if(!cams||!cams.length) return null;
  if(state.pinnedCam) return cams.find(c=>c.id===state.pinnedCam)||cams[0];
  return cams[state.camIndex % cams.length];
}
function ensureOvCam(){
  const v=$('overview-camera-view'); if(!v) return;
  if(!state.ovCam){ const c=document.createElement('canvas'); c.width=640; c.height=360; state.ovCam=c; state.ovCtx=c.getContext('2d'); }
  if(!v.contains(state.ovCam)){ v.innerHTML=''; v.appendChild(state.ovCam); }
}
function ensureThermal(){
  if(!state.thermal){ state.thermal=$('thermal-map'); if(!state.thermal) return; state.thermal.width=720; state.thermal.height=300; state.thermalCtx=state.thermal.getContext('2d'); }
}
function drawOvCam(force){
  if(!state.data||!state.ovCam) return;
  const cam=curCam(); if(!cam) return;
  const ctx=state.ovCtx,cv=state.ovCam;
  ctx.fillStyle='#eef1f6'; ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='rgba(0,143,199,.18)';
  for(let i=0;i<cv.width;i+=40){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,cv.height);ctx.stroke();}
  drawShelves(ctx,cam.zone,cv);
  const zone=cam.zone;
  const ppl=(state.data.shopper.people||[]).filter(p=>p.zone===zone).slice(0,10);
  ppl.forEach((p,i)=>{ const x=(p.x/100)*cv.width,y=(p.y/100)*cv.height;
    ctx.fillStyle='#ff3b30'; ctx.beginPath(); ctx.arc(x,y,6,0,7); ctx.fill();
    ctx.strokeStyle='rgba(0,255,136,.85)'; ctx.strokeRect(x-12,y-18,24,36);
    ctx.fillStyle='#00ff88'; ctx.font='9px monospace'; ctx.fillText((p.behavior||'P').slice(0,4)+' '+(i+1),x-10,y-22); });
  const live=(state.data.shopper.live_detections||{})[cam.id];
  $('ov-cam-name').textContent='• '+cam.name;
  $('ov-cam-res').textContent=cam.res+' • '+cam.fps+' FPS';
  $('ov-cam-zone').textContent='Zone: '+zone;
  $('ov-cam-people').textContent=(live?live.count:ppl.length)+' detected';
  $('ov-cam-live').textContent=live?'● YOLO LIVE':'○ simulated';
  ctx.fillStyle='rgba(0,0,0,.55)'; ctx.fillRect(0,cv.height-22,cv.width,22);
  ctx.fillStyle='#00ff88'; ctx.font='11px monospace';
  ctx.fillText(`● REC ${cam.id} ${new Date().toLocaleTimeString()}`,8,cv.height-7);
  ctx.strokeStyle='#00ff88'; ctx.strokeRect(0,0,cv.width,cv.height);
}
function drawShelves(ctx,zone,cv){
  const w=cv.width,h=cv.height; ctx.lineWidth=3; ctx.strokeStyle='rgba(15,35,70,.35)';
  if(zone==='grocery') for(let i=0;i<4;i++){ctx.beginPath();ctx.moveTo(w*.05,h*(.15+i*.2));ctx.lineTo(w*.95,h*(.15+i*.2));ctx.stroke();}
  else if(zone==='checkout'){ ctx.fillStyle='rgba(0,212,255,.15)'; for(let i=0;i<5;i++){ctx.fillRect(w*(.05+i*.19),h*.6,w*.15,h*.15);} }
  else if(zone && zone.startsWith('promo')){ ctx.fillStyle='rgba(255,136,255,.25)';            ctx.fillRect(w*.1,h*.2,w*.8,h*.6); ctx.fillStyle='#b020c0'; ctx.font='12px sans-serif'; ctx.fillText('PROMO DISPLAY',w*.35,h*.5); }
  else { ctx.strokeStyle='rgba(0,212,255,.35)'; for(let i=0;i<3;i++){ctx.strokeRect(w*(.08+i*.3),h*.15,w*.2,h*.7);} }
}

// ---------- thermal (zone-wise, 3s) ----------
function thermalColor(t){ t=Math.max(0,Math.min(1,t));
  if(t<.33){const f=t/.33;return `rgb(0,${Math.round(50+120*f)},${Math.round(255-215*f)})`;}
  if(t<.66){const f=(t-.33)/.33;return `rgb(${Math.round(255*f)},${Math.round(170+30*f)},${Math.round(40-40*f)})`;}
  const f=(t-.66)/.34; return `rgb(255,${Math.round(200-200*f)},0)`; }
function smoothGrid(d,R,C){ const o=d.map(r=>r.slice());
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){let s=0,w=0;
    for(let dr=-2;dr<=2;dr++)for(let dc=-2;dc<=2;dc++){const nr=r+dr,nc=c+dc;
      if(nr>=0&&nr<R&&nc>=0&&nc<C){const wt=1-(Math.abs(dr)+Math.abs(dc))/5;s+=d[nr][nc]*wt;w+=wt;}}
    o[r][c]=s/Math.max(w,.0001);} return o; }
function drawThermal(){
  if(!state.data||!state.thermal) return;
  const s=state.data.shopper,ctx=state.thermalCtx,cv=state.thermal,layout=s.layout;
  const GC=36,GR=15;
  const den=Array.from({length:GR},()=>new Array(GC).fill(0)); let mx=.0001;
  (s.people||[]).forEach(p=>{const gx=Math.floor(p.x/100*GC),gy=Math.floor(p.y/100*GR);
    if(gx>=0&&gx<GC&&gy>=0&&gy<GR){den[gy][gx]++; mx=Math.max(mx,den[gy][gx]);}});
  for(const [k,z] of Object.entries(layout)){ if(z.type==='storage')continue;
    const occ=s.zone_occupancy[k]||0;
    const gx=Math.floor(z.x/100*GC),gy=Math.floor(z.y/100*GR),gw=Math.max(1,Math.floor(z.width/100*GC)),gh=Math.max(1,Math.floor(z.height/100*GR));
    for(let r=gy;r<gy+gh;r++)for(let c=gx;c<gx+gw;c++)if(r>=0&&r<GR&&c>=0&&c<GC){den[r][c]+=occ*.08; mx=Math.max(mx,den[r][c]);} }
  const bl=smoothGrid(den,GR,GC),cw=cv.width/GC,ch=cv.height/GR;
  for(let r=0;r<GR;r++)for(let c=0;c<GC;c++){ctx.fillStyle=thermalColor(bl[r][c]/mx);ctx.fillRect(c*cw,r*ch,cw+1,ch+1);}
  ctx.font='bold 10px sans-serif'; ctx.textAlign='center';
  for(const [k,z] of Object.entries(layout)){ const x=(z.x+z.width/2)/100*cv.width,y=(z.y+z.height/2)/100*cv.height;
    ctx.fillStyle='rgba(0,0,0,.6)'; ctx.fillRect(x-48,y-11,96,17);
    ctx.fillStyle=z.type==='promo'?'#ff88ff':'#fff'; ctx.fillText(z.label,x,y+2); }
  ctx.textAlign='left';
  const el=$('thermal-zone-stats');
  if(el) el.innerHTML=Object.entries(layout).filter(([k])=>k!=='storage').map(([k,z])=>{
    const occ=s.zone_occupancy[k]||0; const lvl=occ>=12?'Crowded':occ>=7?'Busy':occ>=3?'Normal':'Empty';
    const col=occ>=12?'#ff2200':occ>=7?'#ffcc00':occ>=3?'#00cc66':'#4488ff';
    return `<div class="zone-stat-card"><span class="zone-stat-name">${z.label}</span><span class="zone-stat-val" style="color:${col}">${occ} • ${lvl}</span></div>`;}).join('');
}

// ---------- cameras tab ----------
function camerasTab(d){
  const view=$('camera-main-view');
  if(!state.camMain){ const c=document.createElement('canvas'); c.width=640; c.height=360; state.camMain=c; state.camMainCtx=c.getContext('2d'); }
  if(!view.contains(state.camMain)){ view.innerHTML=''; view.appendChild(state.camMain); }
  const list=$('camera-list'); list.innerHTML='';
  d.cameras.forEach(c=>{
    const el=document.createElement('div'); el.className='cam-item'+(curCam()&&curCam().id===c.id?' active':'');
    el.innerHTML=`<div class="cam-thumb"></div><div><div class="cam-name">${c.name}</div><div class="cam-meta">${c.id} • ${c.zone} • ${c.res}</div></div><span class="cam-status ${c.status}">${c.status}</span>`;
    el.onclick=()=>{ state.pinnedCam = state.pinnedCam===c.id?null:c.id; render(state.data); };
    list.appendChild(el);
  });
  drawCamMain(true);
  const live=d.shopper.live_detections||{};
  $('detection-stats').innerHTML=Object.keys(live).length?Object.entries(live).map(([k,v])=>
    `<div class="det-row"><span class="det-label">● ${k} YOLO live</span><span class="det-count">${v.count}</span></div>`).join('')
    :'<div class="det-row"><span class="det-label">No live YOLO feed — showing simulation. Run detection_bridge.py</span></div>';
}
function drawCamMain(force){
  const cam=curCam(); if(!cam||!state.camMain) return;
  const ctx=state.camMainCtx,cv=state.camMain;
  ctx.fillStyle='#eef1f6'; ctx.fillRect(0,0,cv.width,cv.height);
  drawShelves(ctx,cam.zone,cv);
  const ppl=(state.data.shopper.people||[]).filter(p=>p.zone===cam.zone).slice(0,12);
  ppl.forEach((p,i)=>{const x=p.x/100*cv.width,y=p.y/100*cv.height;
    ctx.fillStyle='#ff3b30';ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.fill();
    ctx.strokeStyle='rgba(0,255,136,.8)';ctx.strokeRect(x-12,y-18,24,36);});
  $('active-cam-title').textContent=cam.name+' ('+cam.id+')';
  $('cam-res').textContent=cam.res+' • '+cam.fps+' FPS';
  $('cam-zone').textContent='Zone: '+cam.zone;
  $('cam-people').textContent=ppl.length+' detected';
}

// ---------- big map ----------
function drawBigMap(){
  if(!state.data) return;
  if(!state.mapCanvas){ state.mapCanvas=$('store-map'); if(!state.mapCanvas) return;
    state.mapCanvas.width=900; state.mapCanvas.height=460; state.mapCtx=state.mapCanvas.getContext('2d'); }
  const s=state.data.shopper,ctx=state.mapCtx,cv=state.mapCanvas;
  ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,cv.width,cv.height);
  const GC=36,GR=18,den=Array.from({length:GR},()=>new Array(GC).fill(0)); let mx=.0001;
  (s.people||[]).forEach(p=>{const gx=Math.floor(p.x/100*GC),gy=Math.floor(p.y/100*GR);
    if(gx>=0&&gx<GC&&gy>=0&&gy<GR){den[gy][gx]++;mx=Math.max(mx,den[gy][gx]);}});
  for(const [k,z] of Object.entries(s.layout)){ if(z.type==='storage')continue;
    const occ=s.zone_occupancy[k]||0,gx=Math.floor(z.x/100*GC),gy=Math.floor(z.y/100*GR),
    gw=Math.max(1,Math.floor(z.width/100*GC)),gh=Math.max(1,Math.floor(z.height/100*GR));
    for(let r=gy;r<gy+gh;r++)for(let c=gx;c<gx+gw;c++)if(r>=0&&r<GR&&c>=0&&c<GC){den[r][c]+=occ*.08;mx=Math.max(mx,den[r][c]);}}
  const bl=smoothGrid(den,GR,GC),cw=cv.width/GC,ch=cv.height/GR;
  for(let r=0;r<GR;r++)for(let c=0;c<GC;c++){ctx.fillStyle=thermalColor(bl[r][c]/mx);ctx.fillRect(c*cw,r*ch,cw+1,ch+1);}
  for(const [k,z] of Object.entries(s.layout)){
    const x=z.x/100*cv.width,y=z.y/100*cv.height,w=z.width/100*cv.width,h=z.height/100*cv.height;
    ctx.strokeStyle=z.type==='promo'?'#c026d3':'rgba(15,35,70,.45)'; ctx.lineWidth=z.type==='promo'?3:2;
    if(z.type==='promo') ctx.setLineDash([6,4]); else ctx.setLineDash([]);
    ctx.strokeRect(x,y,w,h); ctx.setLineDash([]);
    ctx.fillStyle='rgba(0,0,0,.6)'; ctx.fillRect(x+w/2-60,y+h/2-14,120,26);
    ctx.fillStyle=z.type==='promo'?'#ff88ff':'#fff'; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
    ctx.fillText(z.label,x+w/2,y+h/2-1);
    ctx.fillStyle='rgba(15,35,70,.8)'; ctx.font='9px sans-serif';
    ctx.fillText(`${s.zone_occupancy[k]||0} ppl • ${(s.dwell_times[k]||0).toFixed(1)}m`,x+w/2,y+h/2+12); ctx.textAlign='left'; }
  $('map-zone-details').innerHTML=Object.entries(s.layout).filter(([k])=>k!=='storage').map(([k,z])=>
    `<div class="list-item"><span class="name">${z.label}</span><span class="val">${s.zone_occupancy[k]||0} ppl</span><span class="val" style="color:#00ff88">${(s.dwell_times[k]||0).toFixed(1)}m</span></div>`).join('');
  $('map-zone-stats').innerHTML=Object.entries(s.layout).filter(([k])=>k!=='storage').slice(0,6).map(([k,z])=>
    `<div class="zone-stat-card"><span class="zone-stat-name">${z.label}</span><span class="zone-stat-val">${s.zone_occupancy[k]||0}</span></div>`).join('');
}

// ---------- promos ----------
let promoChart=null;
function promosTab(d){
  $('promo-grid').innerHTML=d.promotions.promotions.map(p=>
    `<div class="model-card ${p.best?'promo-best':''}"><div class="model-header"><span class="model-name">${p.best?'⭐ BEST — ':''}${p.name}</span><span class="model-status active">${p.discount}</span></div>
    <p class="model-desc">${p.start} → ${p.end} • Zone footfall ${p.footfall}, dwell ${p.dwell_min} min</p>
    <div class="model-meta"><span>Footfall <strong>${p.footfall}</strong></span><span>Dwell <strong>${p.dwell_min}m</strong></span><span>Conv <strong>${(p.conversion_rate*100).toFixed(1)}%</strong></span><span>Revenue <strong>₹${fmtN(p.revenue)}</strong></span></div>
    <div class="model-accuracy-bar"><div class="model-accuracy-fill" style="width:${Math.min(100,p.conversion_rate*300)}%"></div></div></div>`).join('');
  const cv=$('promo-chart');
  if(cv){ const ps=d.promotions.promotions;
    if(promoChart){promoChart.data.labels=ps.map(p=>p.name);promoChart.data.datasets[0].data=ps.map(p=>p.dwell_min);promoChart.data.datasets[1].data=ps.map(p=>p.footfall);promoChart.update('none');}
    else promoChart=new Chart(cv.getContext('2d'),{type:'bar',data:{labels:ps.map(p=>p.name),datasets:[
      {label:'Dwell (min)',data:ps.map(p=>p.dwell_min),backgroundColor:'rgba(255,136,255,.7)'},
      {label:'Footfall',data:ps.map(p=>p.footfall),backgroundColor:'rgba(0,212,255,.7)',yAxisID:'y2'}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8a94a8'}}},
      scales:{x:{ticks:{color:'#8a94a8'}},y:{ticks:{color:'#8a94a8'},beginAtZero:true},y2:{position:'right',ticks:{color:'#00d4ff'},grid:{drawOnChartArea:false},beginAtZero:true}}}});
  }
}

// ---------- inventory (OUT_OF_STOCK wording) ----------
function invEvents(){
  $('btn-add-product').onclick=()=>openModal();
  $('inventory-search').oninput=e=>invTable($('inv-filter').value,e.target.value);
  $('inv-filter').onchange=e=>invTable(e.target.value);
  $('product-form').onsubmit=saveProduct;
  $('btn-bulk-restock').onclick=async()=>{ for(const it of state.data.inventory.items) if(it.stock_status!=='IN_STOCK') await apiStore('/api/simulate-restock/'+it.id); refreshAll(); };
}
function invTable(filter='all',search=''){
  const inv=state.data.inventory; if(!inv) return;
  $('inv-total').textContent=inv.total_products; $('inv-value').textContent='₹'+fmtN(inv.total_inventory_value);
  $('inv-oos').textContent=inv.out_of_stock_count; $('inv-low').textContent=inv.low_count; $('inv-ok').textContent=inv.in_stock_count;
  let items=[...inv.items];
  if(filter!=='all') items=items.filter(i=>i.stock_status===filter);
  if(search) items=items.filter(i=>i.name.toLowerCase().includes(search.toLowerCase()));
  const pill=s=>s==='OUT_OF_STOCK'?'OUT OF STOCK':s==='LOW'?'LOW':'IN STOCK';
  $('inventory-body').innerHTML=items.map(it=>
    `<tr><td><strong>${it.name}</strong><br><small style="color:#5a6478">${it.barcode||''}</small></td><td>${it.category}</td><td>${it.shelf}</td><td>₹${it.price.toLocaleString('en-IN')}</td>
    <td><div style="display:flex;align-items:center;gap:8px"><div class="progress-bar"><div class="progress-fill ${it.stock_status==='OUT_OF_STOCK'?'critical':it.stock_status.toLowerCase()}" style="width:${it.stock_percentage}%"></div></div><small>${it.current_stock}/${it.max_stock}</small></div></td>
    <td><span class="status-pill ${it.stock_status==='IN_STOCK'?'OK':it.stock_status==='LOW'?'LOW':'CRITICAL'}">${pill(it.stock_status)}</span></td>
    <td><button class="action-btn edit" onclick="editProduct(${it.id})">Edit</button><button class="action-btn restock" onclick="restockProduct(${it.id})">Restock</button><button class="action-btn delete" onclick="deleteProduct(${it.id})">Del</button></td></tr>`).join('');
}
function openModal(p=null){
  $('modal-title').textContent=p?'Edit Product':'Add Product';
  if(p){$('prod-id').value=p.id;$('prod-name').value=p.name;$('prod-category').value=p.category;$('prod-shelf').value=p.shelf;$('prod-price').value=p.price;$('prod-maxstock').value=p.max_stock;$('prod-stock').value=p.current_stock;}
  else{$('product-form').reset();$('prod-id').value='';}
  $('product-modal').classList.add('active');
}
function closeModal(){$('product-modal').classList.remove('active');}
async function saveProduct(e){e.preventDefault();
  const id=$('prod-id').value;
  const body={name:$('prod-name').value,category:$('prod-category').value,shelf:$('prod-shelf').value,price:parseFloat($('prod-price').value),max_stock:parseInt($('prod-maxstock').value||50),current_stock:parseInt($('prod-stock').value||0)};
  if(id) await fetch('/api/inventory/'+id+'?store_id='+state.storeId,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  else await fetch('/api/inventory?store_id='+state.storeId,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  closeModal(); refreshAll();
}
async function editProduct(id){const p=state.data.inventory.items.find(i=>i.id===id); if(p) openModal(p);}
async function restockProduct(id){await apiStore('/api/simulate-restock/'+id); refreshAll();}
async function deleteProduct(id){if(confirm('Delete?')){await fetch('/api/inventory/'+id+'?store_id='+state.storeId,{method:'DELETE'});refreshAll();}}

// ---------- counters ----------
function queuesTab(d){
  const q=d.queue;
  $('q-total-waiting').textContent=q.total_waiting;
  $('q-open-counters').textContent=q.open_counters+'/'+q.max_counters;
  $('q-avg-wait').textContent=q.average_wait_time_min+'m';
  $('q-avg-bill').textContent=q.average_billing_time_min+'m';
  $('queue-recommendation-text').textContent=q.auto_action+` • sales ${q.sales_per_min}/min`;
  $('queue-visual').innerHTML='<div style="position:relative;width:100%;height:100%">'+q.counters.map((c,i)=>{
    const left=4+i*(92/q.counters.length);
    const col=c.status!=='open'?'#333':c.queue_length>=6?'#ffcc00':'#00d4ff';
    let ppl=''; for(let j=0;j<Math.min(c.queue_length,8);j++) ppl+=`<div style="width:8px;height:14px;background:${col};border-radius:3px"></div>`;
    return `<div class="counter-block" style="left:${left}%;border-color:${col}"><strong>C${c.counter_id}</strong><br><small>${c.status}</small><br><small>${c.queue_length} q</small><br><small>~${c.avg_wait_time_min}m</small></div><div style="position:absolute;left:${left}%;bottom:56px;display:flex;gap:3px;width:60px;flex-wrap:wrap;justify-content:center">${ppl}</div>`;
  }).join('')+'</div>';
  $('counter-details').innerHTML=q.counters.map(c=>
    `<div class="list-item"><span class="name">C${c.counter_id} ${c.status}</span><span class="val">${c.queue_length} wait</span><span class="val" style="color:#00d4ff">wait ${c.avg_wait_time_min}m</span><span class="val" style="color:#00ff88">bill ${c.avg_billing_time_min}m</span></div>`).join('');
}

// ---------- reports ----------
async function previewReport(t){
  const d=await apiStore('/api/reports/'+t);
  $('report-preview-title').textContent=t+' — '+d.count+' rows ('+state.storeId+')';
  const rows=d.rows||[];
  $('report-head').innerHTML=rows.length?'<tr>'+Object.keys(rows[0]).map(k=>`<th>${k}</th>`).join('')+'</tr>':'';
  $('report-body').innerHTML=rows.slice(0,20).map(r=>'<tr>'+Object.values(r).map(v=>`<td>${v}</td>`).join('')+'</tr>').join('');
}
function downloadReport(t,fmt){ window.open(`/api/reports/${t}?store_id=${state.storeId}&fmt=${fmt}`,'_blank'); }

// ---------- models / alerts ----------
function modelsTab(d){
  $('models-grid').innerHTML=d.ai_models.map(m=>
    `<div class="model-card"><div class="model-header"><span class="model-name">${m.name}</span><span class="model-status ${m.status}">${m.status}</span></div><p class="model-desc">${m.description}</p><div class="model-meta"><span>Acc <strong>${m.accuracy}%</strong></span><span><strong>${m.fps} FPS</strong></span><span>${m.framework}</span></div><div class="model-accuracy-bar"><div class="model-accuracy-fill" style="width:${m.accuracy}%"></div></div><div class="model-target">${m.target}</div></div>`).join('');
}
function alertsTab(d){
  $('alerts-full-list').innerHTML=d.alerts.slice().reverse().map(a=>
    `<div class="alert-item ${a.severity}"><span class="alert-icon">${a.severity==='critical'?'🔴':'⚠️'}</span><div class="alert-content"><div class="alert-message">${a.message}</div><div class="alert-time">${a.type} • ${new Date(a.timestamp).toLocaleTimeString('en-IN')}</div></div></div>`).join('')||'<div style="color:#5a6478;padding:20px">No alerts</div>';
}
