let _vid = 0;
class Value {
  constructor(data, children=[], op='', label=''){
    this.data = data; this.grad = 0;
    this._bwd = ()=>{};
    this._prev = new Set(children);
    this._op = op; this.label = label;
    this.id = _vid++;
  }
  add(o){ o=ov(o); const r=new Value(this.data+o.data,[this,o],'+'); r._bwd=()=>{this.grad+=r.grad; o.grad+=r.grad}; return r }
  mul(o){ o=ov(o); const r=new Value(this.data*o.data,[this,o],'×'); r._bwd=()=>{this.grad+=o.data*r.grad; o.grad+=this.data*r.grad}; return r }
  pow(n){ const r=new Value(Math.pow(this.data,n),[this],`^${n}`); r._bwd=()=>{this.grad+=n*Math.pow(this.data,n-1)*r.grad}; return r }
  relu(){ const r=new Value(this.data>0?this.data:0,[this],'ReLU'); r._bwd=()=>{this.grad+=(r.data>0?1:0)*r.grad}; return r }
  neg(){ return this.mul(new Value(-1)) }
  sub(o){ return this.add(ov(o).neg()) }
  div(o){ return this.mul(ov(o).pow(-1)) }
  backward(){
    const topo=[], seen=new Set();
    const build=v=>{ if(!seen.has(v)){ seen.add(v); for(const c of v._prev) build(c); topo.push(v) } };
    build(this); this.grad=1;
    for(const v of topo.reverse()) v._bwd();
    return topo;
  }
}
const ov = x => x instanceof Value ? x : new Value(x);

class Neuron {
  constructor(nin, nonlin=true){
    this.w = Array.from({length:nin},()=>new Value((Math.random()*2-1)*.8));
    this.b = new Value(0); this.nonlin=nonlin;
  }
  fwd(x){ let a=this.b; for(let i=0;i<this.w.length;i++) a=a.add(this.w[i].mul(x[i])); return this.nonlin?a.relu():a }
  params(){ return [...this.w,this.b] }
  zeroGrad(){ this.params().forEach(p=>p.grad=0) }
}

class Layer {
  constructor(nin,nout,nonlin=true){ this.neurons=Array.from({length:nout},()=>new Neuron(nin,nonlin)) }
  fwd(x){ const o=this.neurons.map(n=>n.fwd(x)); return o.length===1?o[0]:o }
  params(){ return this.neurons.flatMap(n=>n.params()) }
  zeroGrad(){ this.neurons.forEach(n=>n.zeroGrad()) }
}

class MLP {
  constructor(nin,nouts){
    const sz=[nin,...nouts];
    this.layers=sz.slice(0,-1).map((s,i)=>new Layer(s,sz[i+1],i!==nouts.length-1));
    this.nin=nin; this.nouts=nouts;
  }
  fwd(x){ let o=x.map(v=>ov(v)); for(const l of this.layers) o=l.fwd(o); return o }
  params(){ return this.layers.flatMap(l=>l.params()) }
  zeroGrad(){ this.layers.forEach(l=>l.zeroGrad()) }
}

function fastFwd(net, x){
  let h = [...x];
  for(const layer of net.layers){
    h = layer.neurons.map(n=>{
      let a = n.b.data;
      for(let i=0;i<n.w.length;i++) a += n.w[i].data * h[i];
      return n.nonlin ? Math.max(0,a) : a;
    });
  }
  return Array.isArray(h) ? h[0] : h;
}


const ns='http://www.w3.org/2000/svg';
const mk=(tag,a={})=>{ const e=document.createElementNS(ns,tag); for(const[k,v] of Object.entries(a)) e.setAttribute(k,v); return e };
const fmt=(v,d=3)=>typeof v==='number'?v.toFixed(d):'?';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const lerp=(a,b,t)=>a+(b-a)*t;
const ease=t=>t<.5?2*t*t:-1+(4-2*t)*t;

function bezier(x1,y1,x2,y2){
  const mx=(x1+x2)/2;
  return `M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`;
}


const tip = document.getElementById('tip');
function showTip(e, label, data, grad, hasVals){
  document.getElementById('tl').textContent = label;
  document.getElementById('td').textContent = hasVals ? `data: ${fmt(data)}` : 'data: ?';
  document.getElementById('tg').textContent = (hasVals && grad!==0) ? `grad: ${fmt(grad)}` : '';
  tip.classList.add('on');
  moveTip(e);
}
function moveTip(e){ tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-8)+'px' }
function hideTip(){ tip.classList.remove('on') }



let cg = null, cgPhase = 0;

function buildCG(){
  _vid = 0;
  const a=new Value(2.0,'','','a');
  const b=new Value(-3.0,'','','b');
  const c=new Value(4.0,'','','c');
  const d=a.add(b); d.label='d';
  const L=d.mul(c); L.label='L';
  return {a,b,c,d,L};
}

const CG = {
  W:660, H:300,
  nodes:{
    a:  {x:30,  y:50  },
    b:  {x:30,  y:140 },
    c:  {x:30,  y:230 },
    d:  {x:300, y:95  },
    L:  {x:510, y:165 },
  },
  ops:{
    plus: {x:195, y:118},
    mul:  {x:420, y:185},
  },
  BW:86, BH:44,
};

function renderCG(){
  const svg = document.getElementById('cg-svg');
  svg.innerHTML='';
  if(!cg) cg = buildCG();

  const phase = cgPhase;
  const hasF = phase >= 1;
  const hasB = phase >= 2;
  const C = CG;

  const bc = key => ({ x: C.nodes[key].x + C.BW/2, y: C.nodes[key].y + C.BH/2 });
  const oc = key => C.ops[key];


  const edges = [
    { from: {x:C.nodes.a.x+C.BW, y:C.nodes.a.y+C.BH/2}, to: oc('plus') },
    { from: {x:C.nodes.b.x+C.BW, y:C.nodes.b.y+C.BH/2}, to: oc('plus') },
    { from: oc('plus'), to: {x:C.nodes.d.x, y:C.nodes.d.y+C.BH/2} },
    { from: {x:C.nodes.d.x+C.BW, y:C.nodes.d.y+C.BH/2}, to: oc('mul') },
    { from: {x:C.nodes.c.x+C.BW, y:C.nodes.c.y+C.BH/2}, to: oc('mul') },
    { from: oc('mul'), to: {x:C.nodes.L.x, y:C.nodes.L.y+C.BH/2} },
  ];

  edges.forEach(({from,to},i)=>{
    const isBack = hasB;
    const color = isBack ? '#ffd86e22' : (hasF ? '#58c4dd22' : '#1e1e1e');
    const e = mk('path',{
      d: bezier(from.x,from.y,to.x,to.y),
      stroke: color, 'stroke-width':'1.5', fill:'none',
      'stroke-dasharray': isBack ? '3 3' : 'none',
    });
    svg.appendChild(e);
  });

  [['plus','+'],['mul','×']].forEach(([key,sym])=>{
    const {x,y} = C.ops[key];
    const g = mk('g');
    g.appendChild(mk('circle',{cx:x,cy:y,r:14,fill:'#1a1a1a',stroke:hasF?'#333':'#1c1c1c','stroke-width':'1'}));
    const t = mk('text',{x,y:y+4,'text-anchor':'middle',fill:hasF?'#888':'#333','font-size':'13','font-family':'SF Mono,monospace'});
    t.textContent=sym; g.appendChild(t);
    svg.appendChild(g);
  });

  const nodeKeys = ['a','b','c','d','L'];
  nodeKeys.forEach(key=>{
    const {x,y} = C.nodes[key];
    const node = cg[key];
    const isLeaf = key==='a'||key==='b'||key==='c';
    const g = mk('g',{style:'cursor:pointer'});

    if(hasB && Math.abs(node.grad)>0.01){
      const glow = mk('rect',{x:x-2,y:y-2,width:C.BW+4,height:C.BH+4,rx:7,fill:'none',stroke:'rgba(255,216,110,0.15)','stroke-width':'1'});
      g.appendChild(glow);
    }

    const boxStroke = hasB ? '#ffd86e44' : (hasF ? '#58c4dd33' : '#222');
    const box = mk('rect',{x,y,width:C.BW,height:C.BH,rx:5,fill:'#111',stroke:boxStroke,'stroke-width':'1'});
    g.appendChild(box);

    const lbl = mk('text',{x:x+7,y:y+12,fill:'#444','font-size':'9','font-family':'SF Mono,monospace'});
    lbl.textContent = key;
    g.appendChild(lbl);

    const dv = mk('text',{x:x+7,y:y+27,fill:hasF?(isLeaf?'#5dd47c':'#58c4dd'):'#333','font-size':'11','font-family':'SF Mono,monospace','font-weight':'500'});
    dv.textContent = hasF ? fmt(node.data) : '--';
    g.appendChild(dv);

    if(hasB){
      const gv = mk('text',{x:x+C.BW-7,y:y+38,'text-anchor':'end',fill:'#ffd86e','font-size':'9.5','font-family':'SF Mono,monospace'});
      gv.textContent = `∇ ${fmt(node.grad)}`;
      g.appendChild(gv);
    }

    if(!isLeaf && node._op){
      const bx = mk('text',{x:x+C.BW-8,y:y+12,'text-anchor':'end',fill:'#333','font-size':'8','font-family':'SF Mono,monospace'});
      bx.textContent = node._op;
      g.appendChild(bx);
    }

    g.addEventListener('mouseenter',e=>{
      box.setAttribute('stroke',hasB?'#ffd86eaa':'#58c4ddaa');
      showTip(e,key,node.data,node.grad,hasF);
    });
    g.addEventListener('mousemove',moveTip);
    g.addEventListener('mouseleave',()=>{ box.setAttribute('stroke',boxStroke); hideTip() });

    svg.appendChild(g);
  });

  const hint = document.getElementById('cg-hint');
  if(phase===0) hint.textContent='click "forward pass" to begin'; hint.className='hint';
  if(phase===1){ hint.textContent='forward pass complete, values shown in blue'; hint.className='hint ok'; }
  if(phase===2){ hint.textContent='backward pass complete, gradients shown in gold'; hint.className='hint ok'; }
}

function cgForward(){ cg=buildCG(); cgPhase=1; renderCG() }
function cgBackward(){
  if(!cg) cg=buildCG();
  if(cgPhase<1) cgForward();
  cg.L.backward(); cgPhase=2; renderCG();
}
function cgReset(){ cg=buildCG(); cgPhase=0; renderCG() }

renderCG();

let nNet = null, nOut = null, nPhase = 0;
const NX = [ 1.5, -0.5 ];

function makeNeuron(){
  const n = new Neuron(2,true);
  n.w[0].data=0.6; n.w[1].data=-0.4; n.b.data=0.1;
  return n;
}

function renderNeuron(){
  const svg = document.getElementById('n-svg');
  svg.innerHTML='';
  if(!nNet) nNet=makeNeuron();

  const W=660, H=260;
  const hasF = nPhase>=1, hasB = nPhase>=2;

  const x1={x:60,y:80}, x2={x:60,y:180};
  const sigma={x:280,y:130};
  const relu={x:430,y:130};
  const out={x:590,y:130};
  const R=26;

  const paths=[
    {f:{x:x1.x+R,y:x1.y}, t:{x:sigma.x-R,y:sigma.y}, weight:nNet.w[0], xlabel:'w₁'},
    {f:{x:x2.x+R,y:x2.y}, t:{x:sigma.x-R,y:sigma.y}, weight:nNet.w[1], xlabel:'w₂'},
    {f:{x:sigma.x+R,y:sigma.y}, t:{x:relu.x-R,y:relu.y}, weight:null, xlabel:''},
    {f:{x:relu.x+R,y:relu.y}, t:{x:out.x-R,y:out.y}, weight:null, xlabel:''},
  ];

  paths.forEach(({f,t,weight,xlabel},i)=>{
    const activeEdge = hasF;
    const color = activeEdge ? (i<2?'#58c4dd44':'#5dd47c44') : '#1e1e1e';
    const p = mk('path',{d:bezier(f.x,f.y,t.x,t.y),stroke:color,'stroke-width':'1.5',fill:'none'});
    svg.appendChild(p);

    if(xlabel){
      const mx=(f.x+t.x)/2, my=(f.y+t.y)/2 - 14;
      const wt = mk('text',{x:mx,y:my,'text-anchor':'middle',fill:hasF?'#58c4dd88':'#2a2a2a','font-size':'9.5','font-family':'SF Mono,monospace'});
      wt.textContent = `${xlabel}=${fmt(weight.data,2)}`;
      svg.appendChild(wt);
      if(hasB && weight){
        const gt = mk('text',{x:mx,y:my+13,'text-anchor':'middle',fill:'#ffd86eaa','font-size':'9','font-family':'SF Mono,monospace'});
        gt.textContent = `∇${fmt(weight.grad,3)}`;
        svg.appendChild(gt);
      }
      const wr=22, hr=hasB?28:16;
      const hl = mk('rect',{x:mx-wr,y:my-12,width:wr*2,height:hr,rx:3,fill:'transparent',style:'cursor:pointer'});
      if(weight){
        hl.addEventListener('mouseenter',e=>{ wt.setAttribute('fill','#58c4ddcc'); showTip(e,xlabel,weight.data,weight.grad,hasF) });
        hl.addEventListener('mousemove',moveTip);
        hl.addEventListener('mouseleave',()=>{ wt.setAttribute('fill',hasF?'#58c4dd88':'#2a2a2a'); hideTip() });
      }
      svg.appendChild(hl);
    }
  });

  const bx = mk('text',{x:sigma.x,y:sigma.y-R-16,'text-anchor':'middle',fill:hasF?'#888':'#2a2a2a','font-size':'9.5','font-family':'SF Mono,monospace'});
  bx.textContent = `b = ${fmt(nNet.b.data,2)}`;
  svg.appendChild(bx);
  if(hasB){
    const bg = mk('text',{x:sigma.x,y:sigma.y-R-4,'text-anchor':'middle',fill:'#ffd86eaa','font-size':'9','font-family':'SF Mono,monospace'});
    bg.textContent = `∇b = ${fmt(nNet.b.grad,3)}`;
    svg.appendChild(bg);
  }

  const circles=[
    {p:x1, label:'x₁', val:NX[0], color:hasF?'#5dd47c':'#222', node:null},
    {p:x2, label:'x₂', val:NX[1], color:hasF?'#5dd47c':'#222', node:null},
    {p:sigma, label:'Σ+b', val:nOut?[...nOut._prev][0]?.data:null, color:hasF?'#b09de8':'#222', node:nOut?[...nOut._prev][0]:null},
    {p:relu, label:'ReLU', val:nOut?.data, color:(hasF)?'#58c4dd':'#222', node:nOut},
    {p:out, label:'out', val:nOut?.data, color:(hasF)?'#58c4dd':'#222', node:nOut},
  ];

  circles.forEach(({p,label,val,color,node})=>{
    const g = mk('g',{transform:`translate(${p.x},${p.y})`,style:'cursor:pointer'});
    const c = mk('circle',{cx:0,cy:0,r:R,fill:'#111',stroke:color,'stroke-width':'1.5'});
    g.appendChild(c);

    const lt = mk('text',{x:0,y:-10,'text-anchor':'middle',fill:'#444','font-size':'8.5','font-family':'SF Mono,monospace'});
    lt.textContent=label; g.appendChild(lt);

    const vt = mk('text',{x:0,y:6,'text-anchor':'middle',fill:color,'font-size':'10.5','font-family':'SF Mono,monospace','font-weight':'500'});
    vt.textContent = (hasF && val!==null && val!==undefined) ? fmt(val,3) : '--';
    g.appendChild(vt);

    if(hasB && node && node.grad!==0){
      const gt = mk('text',{x:0,y:19,'text-anchor':'middle',fill:'#ffd86e','font-size':'8.5','font-family':'SF Mono,monospace'});
      gt.textContent=`∇${fmt(node.grad,2)}`; g.appendChild(gt);
    }

    const hasVal = hasF && val!==null && val!==undefined;
    g.addEventListener('mouseenter',e=>{ c.setAttribute('stroke',hasB?'#ffd86e':'#58c4ddcc'); showTip(e,label,val,node?.grad??0,hasVal) });
    g.addEventListener('mousemove',moveTip);
    g.addEventListener('mouseleave',()=>{ c.setAttribute('stroke',color); hideTip() });
    svg.appendChild(g);
  });

  const rvG = mk('g',{transform:`translate(${relu.x-18},${relu.y+R+8})`});
  rvG.appendChild(mk('path',{d:'M0 20 L18 20 L36 0',stroke:'#5dd47c44','stroke-width':'1.5',fill:'none','stroke-linecap':'round'}));
  svg.appendChild(rvG);

  const hint = document.getElementById('n-hint');
  if(nPhase===0) hint.textContent='hover weights and nodes to inspect values'; hint.className='hint';
  if(nPhase===1){ hint.textContent=`output = ${fmt(nOut?.data,4)} (after ReLU)`; hint.className='hint ok'; }
  if(nPhase===2){ hint.textContent=`gradients propagated: ∇w₁=${fmt(nNet.w[0].grad,4)}, ∇w₂=${fmt(nNet.w[1].grad,4)}`; hint.className='hint ok'; }
}

function nForward(){
  nNet=makeNeuron(); _vid=1000;
  const x=NX.map(v=>{ const val=new Value(v); return val });
  nOut=nNet.fwd(x); nPhase=1; renderNeuron();
}
function nBackward(){
  if(nPhase<1) nForward();
  nOut.backward(); nPhase=2; renderNeuron();
}
function nReset(){ nNet=makeNeuron(); nOut=null; nPhase=0; renderNeuron() }

renderNeuron();


let mlp = new MLP(2,[4,4,1]), mlpOut=null, mlpPhase=0;

function renderMLP(){
  const svg = document.getElementById('mlp-svg');
  svg.innerHTML='';

  const W=660, H=340;
  const arch=[2,4,4,1];
  const lx=[80,240,420,580];
  const lcolors=['#58c4dd','#b09de8','#b09de8','#5dd47c'];
  const lnames=['input','hidden','hidden','output'];
  const hasF=mlpPhase>=1, hasB=mlpPhase>=2;
  const R=16;

  const pos = arch.map((n,li)=>{
    const sp = Math.min(68,(H-60)/n);
    const tot=(n-1)*sp;
    return Array.from({length:n},(_,i)=>({x:lx[li],y:H/2-tot/2+i*sp}));
  });

  for(let li=0;li<arch.length-1;li++){
    const fl=pos[li], tl=pos[li+1];
    for(let fi=0;fi<fl.length;fi++){
      for(let ti=0;ti<tl.length;ti++){
        const w = mlp.layers[li]?.neurons?.[ti]?.w?.[fi];
        const wd = w?.data ?? 0;
        let stroke = '#181818';
        let sw = '0.5';
        if(hasF && w){
          const alpha = clamp(Math.abs(wd)*.5+.05,0,.7);
          stroke = wd>0?`rgba(88,196,221,${alpha})`:`rgba(255,107,107,${alpha})`;
          sw = String(clamp(Math.abs(wd)*1.2+.4,0.4,2.5));
        }
        const {x:fx,y:fy}=fl[fi], {x:tx,y:ty}=tl[ti];
        svg.appendChild(mk('line',{x1:fx+R,y1:fy,x2:tx-R,y2:ty,stroke,['stroke-width']:sw}));
      }
    }
  }

  pos.forEach((layer,li)=>{
    const col = lcolors[li];
    layer.forEach((p,ni)=>{
      const g = mk('g',{transform:`translate(${p.x},${p.y})`,style:'cursor:pointer'});
      const stroke = hasF?col:'#222';
      const c = mk('circle',{cx:0,cy:0,r:R,fill:'#111',stroke,['stroke-width']:'1.5'});
      g.appendChild(c);
      const lbl = mk('text',{x:0,y:4,'text-anchor':'middle',fill:hasF?col:'#333','font-size':'8.5','font-family':'SF Mono,monospace'});
      if(li===0) lbl.textContent=`x${ni+1}`;
      else if(li===arch.length-1) lbl.textContent='out';
      else lbl.textContent='h';
      g.appendChild(lbl);

      if(hasB){
        const n = mlp.layers[li-1]?.neurons?.[ni];
        const gv = n ? n.params().reduce((s,p)=>s+Math.abs(p.grad),0)/n.params().length : 0;
        if(gv>0.001){
          const ring = mk('circle',{cx:0,cy:0,r:R+3,fill:'none',stroke:'rgba(255,216,110,0.25)','stroke-width':'1.5'});
          g.insertBefore(ring,c);
        }
      }

      g.addEventListener('mouseenter',e=>{ c.setAttribute('stroke',hasB?'#ffd86e':'#ffffffcc'); });
      g.addEventListener('mouseleave',()=>{ c.setAttribute('stroke',stroke) });
      svg.appendChild(g);
    });
  });

  lx.forEach((x,i)=>{
    const t = mk('text',{x,y:H-8,'text-anchor':'middle',fill:'#2a2a2a','font-size':'8','font-family':'SF Mono,monospace','letter-spacing':'0.08em'});
    t.textContent=lnames[i].toUpperCase();
    svg.appendChild(t);
    const n2 = mk('text',{x,y:H-20,'text-anchor':'middle',fill:'#333','font-size':'8','font-family':'SF Mono,monospace'});
    n2.textContent=`[${arch[i]}]`; svg.appendChild(n2);
  });

  if(hasF){
    const arr = mk('text',{x:W/2,y:20,'text-anchor':'middle',fill:'#333','font-size':'9','font-family':'SF Mono,monospace'});
    arr.textContent = hasB ? '← gradients flow backward' : '→ signal flows forward';
    svg.appendChild(arr);
  }

  const hint = document.getElementById('mlp-hint');
  if(mlpPhase===0){hint.textContent='architecture: 2 → 4 → 4 → 1'; hint.className='hint';}
  if(mlpPhase===1){hint.textContent='forward pass done: edge color = weight sign, thickness = weight magnitude'; hint.className='hint ok';}
  if(mlpPhase===2){hint.textContent='backward pass done: gold rings = high gradient magnitude neurons'; hint.className='hint ok';}
}

function mlpForward(){
  mlp=new MLP(2,[4,4,1]); _vid=5000;
  mlpOut=mlp.fwd([1.0,0.5]); mlpPhase=1; renderMLP();
}
function mlpBackward(){
  if(mlpPhase<1) mlpForward();
  mlpOut.backward(); mlpPhase=2; renderMLP();
}
function mlpReset(){ mlp=new MLP(2,[4,4,1]); mlpOut=null; mlpPhase=0; renderMLP() }

renderMLP();


let trainNet=new MLP(2,[8,8,1]);
let trainStep=0, lossHist=[], isRunning=false, trainTimer=null;

function makeDataset(){
  const X=[], Y=[];
  for(let i=0;i<120;i++){
    const angle=Math.random()*Math.PI*2;
    if(i<60){
      const r=Math.random()*.6+.1;
      X.push([r*Math.cos(angle),r*Math.sin(angle)]); Y.push(1);
    } else {
      const r=Math.random()*.5+.9;
      X.push([r*Math.cos(angle)*1.1,r*Math.sin(angle)*1.1]); Y.push(-1);
    }
  }
  return {X,Y};
}
const DS = makeDataset();

function trainOnce(){
  const {X,Y}=DS;
  _vid = 50000 + trainStep*10000;

  const ypred = X.map(x=>trainNet.fwd(x));
  let total = new Value(0);
  let correct=0;
  Y.forEach((yi,i)=>{
    const margin = new Value(1).add(ypred[i].mul(-yi));
    if(margin.data>0) total=total.add(margin);
    if((ypred[i].data>0&&yi>0)||(ypred[i].data<0&&yi<0)) correct++;
  });
  const dataLoss = total.mul(1/Y.length);

  let reg=new Value(0);
  trainNet.params().forEach(p=>{ reg=reg.add(p.mul(p)) });
  const loss = dataLoss.add(reg.mul(1e-4));

  trainNet.zeroGrad();
  loss.backward();

  const lr=0.05;
  trainNet.params().forEach(p=>{ p.data -= lr*p.grad });

  const acc=(correct/Y.length*100).toFixed(1);
  lossHist.push(loss.data);
  trainStep++;

  document.getElementById('sv').textContent=trainStep;
  document.getElementById('lv').textContent=loss.data.toFixed(4);
  document.getElementById('av').textContent=acc+'%';

  drawLoss(); drawBoundary();
  return loss.data;
}

function drawLoss(){
  const cv=document.getElementById('loss-cv');
  const ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.fillStyle='#0e0e0e'; ctx.fillRect(0,0,W,H);

  ctx.strokeStyle='#161616'; ctx.lineWidth=1;
  for(let i=1;i<4;i++){
    ctx.beginPath(); ctx.moveTo(40,i*H/4); ctx.lineTo(W-10,i*H/4); ctx.stroke();
  }

  if(lossHist.length<2) return;

  const maxL=Math.max(...lossHist.slice(0,5),0.5);
  const pW=W-50, pH=H-30;

  ctx.strokeStyle='#58c4dd'; ctx.lineWidth=1.5;
  ctx.shadowColor='#58c4dd44'; ctx.shadowBlur=6;
  ctx.beginPath();
  lossHist.forEach((l,i)=>{
    const x=40+(i/Math.max(lossHist.length-1,1))*pW;
    const y=10+(1-l/maxL)*pH;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke(); ctx.shadowBlur=0;

  const lx=40+pW, ll=lossHist[lossHist.length-1];
  const ly=10+(1-ll/maxL)*pH;
  ctx.fillStyle='#58c4dd'; ctx.beginPath(); ctx.arc(lx,ly,3.5,0,Math.PI*2); ctx.fill();

  ctx.fillStyle='#333'; ctx.font='9px SF Mono,monospace';
  ctx.fillText('loss',6,14); ctx.fillText(trainStep,W-20,H-4);
}

function drawBoundary(){
  const cv=document.getElementById('bound-cv');
  const ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.fillStyle='#0e0e0e'; ctx.fillRect(0,0,W,H);

  const res=28, cw=W/res, ch=H/res;
  for(let i=0;i<res;i++){
    for(let j=0;j<res;j++){
      const px=(i/res-.5)*4, py=(j/res-.5)*4;
      const v=fastFwd(trainNet,[px,py]);
      const a=clamp(Math.abs(v)*.25,.02,.45);
      ctx.fillStyle=v>0?`rgba(93,212,124,${a})`:`rgba(255,107,107,${a})`;
      ctx.fillRect(i*cw,j*ch,cw+1,ch+1);
    }
  }

  const {X,Y}=DS;
  const sc=W/4;
  X.forEach(([x,y],i)=>{
    const px=W/2+x*sc, py=H/2+y*sc;
    ctx.fillStyle=Y[i]>0?'#5dd47c':'#ff6b6b';
    ctx.beginPath(); ctx.arc(px,py,3,0,Math.PI*2); ctx.fill();
  });
}

function toggleTrain(){
  if(isRunning){
    clearInterval(trainTimer); isRunning=false;
    document.getElementById('train-btn').textContent='▶ resume training';
    document.getElementById('train-btn').className='primary';
  } else {
    isRunning=true;
    document.getElementById('train-btn').textContent='⏸ pause';
    trainTimer=setInterval(()=>{
      const l=trainOnce();
      if(l<0.008||trainStep>=300){
        clearInterval(trainTimer); isRunning=false;
        document.getElementById('train-btn').textContent='✓ converged';
        document.getElementById('lv').style.color='var(--green)';
      }
    },60);
  }
}

function resetTrain(){
  clearInterval(trainTimer); isRunning=false;
  trainNet=new MLP(2,[8,8,1]); lossHist=[]; trainStep=0;
  document.getElementById('train-btn').textContent='▶ start training';
  document.getElementById('train-btn').className='primary';
  document.getElementById('sv').textContent='0';
  document.getElementById('lv').textContent='--';
  document.getElementById('lv').style.color='';
  document.getElementById('av').textContent='--';
  const lc=document.getElementById('loss-cv'); lc.getContext('2d').fillStyle='#0e0e0e'; lc.getContext('2d').fillRect(0,0,lc.width,lc.height);
  drawBoundary();
}

(()=>{ const c=document.getElementById('loss-cv'); c.getContext('2d').fillStyle='#0e0e0e'; c.getContext('2d').fillRect(0,0,c.width,c.height); drawBoundary() })();



const obs = new IntersectionObserver(entries=>{
  entries.forEach(e=>{ if(e.isIntersecting) e.target.classList.add('visible') });
},{threshold:.1});
document.querySelectorAll('section:not(.hero)').forEach(s=>obs.observe(s));
