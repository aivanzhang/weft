"use strict";
((root) => {
  const LIMITS = Object.freeze({events:18001,changes:1024,parts:384,vertices:100000});
  const blockedImports = "\n;globalThis.importScripts=function(){throw new Error('Offline replay forbids imported scripts');};\n";
  const unwrap = snapshot => snapshot?.profile === "shared-exercise-state-v1" ? snapshot.engine_state : snapshot;
  function settingsAt(changes,index) {
    let left=0,right=changes.length;
    while(left<right){const mid=(left+right)>>1;if(changes[mid].after_event_index<=index)left=mid+1;else right=mid;}
    if(!left)throw new Error("Recorded appearance is missing at this event");
    return changes[left-1];
  }
  function compactScene(scene) {
    const value=scene.engine_scene||scene,h=value.human_spec;
    return {profile:value.profile,bounds:value.bounds,dt:value.dt,coordinate_frame:{scale_bridge:value.coordinate_frame?.scale_bridge},
      human_spec:{lengths:h?.lengths,body_frame:h?.body_frame,mobility:{base_height:h?.mobility?.base_height,ground_plane:h?.mobility?.ground_plane}},
      objects:value.objects?.map(object=>{
        const out=Object.fromEntries(["id","shape","role","size","radius","length","color","position","quaternion"].filter(key=>object[key]!==undefined).map(key=>[key,object[key]]));
        if(object.shape==="compound"){
          const parts=object.geometry?.parts??object.parts;
          if(!Array.isArray(parts)||!parts.length||parts.length>32)throw new Error("Avatar compound bound exceeded");
          // Preserve render geometry, not the source's inertia or provenance metadata.
          out.parts=parts.map(part=>Object.fromEntries(["shape","size","radius","length","position","quaternion"].filter(key=>part?.[key]!==undefined).map(key=>[key,part[key]])));
        }
        return out;
      })};
  }
  function describe(event) {
    const snapshot=event.snapshot,state=unwrap(snapshot),metrics=event.exercise_metrics||snapshot.task_metrics;
    const type=event.type==="start"?"Recording start":event.type==="reset"?"Seeded reset":event.type==="simulation_failure"?"Recorded simulation failure":"Recorded action";
    const time=metrics?.simulated_duration_s??state.time_s??0;
    let outcome=`Saved simulation time ${Number(time).toFixed(3)} s`;
    if(metrics){
      const status=metrics.success&&metrics.completed_interval?"passed":metrics.failure||metrics.completed_interval?"not passed":"in progress";
      outcome+=` · Recorded task: ${status}`;
      if(Array.isArray(metrics.failed_gates)&&metrics.failed_gates.length)outcome+=` · ${metrics.failed_gates.join(", ")}`;
    }
    if((event.engine_state_status||snapshot.engine_state_status||"current")!=="current")outcome+=" · Showing last valid geometry after a recorded failure";
    let controller="Controller unspecified; no intent inferred from action values";
    if(event.type==="start"||event.type==="reset")controller="No action applied at this recorded boundary";
    else if(event.controller==="manual")controller="Recorded manual human action";
    else if(event.controller==="coast")controller="Recorded coast command";
    else if(event.controller==="authored")controller="Recorded authored feedback; programmed, not learned";
    else if(event.controller==="learned")controller="Recorded policy"+(/^[0-9a-f]{64}$/.test(event.policy_sha256||"")?` · SHA256 ${event.policy_sha256.slice(0,12)}`:" · identity unavailable");
    return {type,outcome,controller,time};
  }
  function roundPaths(shape) {
    const r=shape.radius,n=16,paths=[];
    if(!(Number.isFinite(r)&&r>0))throw new Error("Invalid recorded round shape");
    if(shape.shape==="sphere"){
      for(const [a,b] of [[0,1],[0,2],[1,2]])paths.push(Array.from({length:n+1},(_,i)=>{const p=[0,0,0];p[a]=r*Math.cos(i*2*Math.PI/n);p[b]=r*Math.sin(i*2*Math.PI/n);return p;}));
    }else{
      const h=shape.length/2;if(!(Number.isFinite(h)&&h>=0))throw new Error("Invalid recorded capsule length");
      for(const axis of [0,1]){
        const path=[];
        for(const offset of [h,-h])for(let i=0;i<=n/2;i++){const theta=i*2*Math.PI/n+(offset<0?Math.PI:0),p=[0,0,offset+r*Math.sin(theta)];p[axis]=r*Math.cos(theta);path.push(p);}
        path.push(path[0]);paths.push(path);
      }
      for(const z of [h,-h])paths.push(Array.from({length:n+1},(_,i)=>[r*Math.cos(i*2*Math.PI/n),r*Math.sin(i*2*Math.PI/n),z]));
    }
    return paths;
  }
  function shapePaths(shape) {
    if(shape.shape==="sphere"||shape.shape==="capsule")return roundPaths(shape);
    if(shape.shape!=="box"||!Array.isArray(shape.size)||shape.size.length!==3||shape.size.some(v=>!Number.isFinite(v)||v<=0))throw new Error("Unsupported recorded equipment shape");
    const corners=Array.from({length:8},(_,i)=>shape.size.map((v,j)=>((i>>j)&1?1:-1)*v/2)),paths=[];
    for(let i=0;i<8;i++)for(let axis=0;axis<3;axis++)if(!(i&(1<<axis)))paths.push([corners[i],corners[i|(1<<axis)]]);
    return paths;
  }
  function poseMatrix(math,position,quaternion) {
    if(!Array.isArray(position)||position.length!==3||position.some(v=>!Number.isFinite(v))||!Array.isArray(quaternion)||quaternion.length!==4||Math.abs(Math.hypot(...quaternion)-1)>1e-6)throw new Error("Invalid recorded equipment transform");
    const [w,x,y,z]=quaternion;return math.transform(position,[x,y,z,w]);
  }
  function prepareOutlines(math,config) {
    let count=0,vertices=0;
    const objects=new Map();
    for(const object of config.objects){
      if(objects.has(object.id))throw new Error("Duplicate recorded equipment identity");
      const parts=object.shape==="compound"?object.parts:[object],paths=[];
      if(!Array.isArray(parts)||!parts.length)throw new Error("Missing recorded equipment parts");
      for(const part of parts){
        if(++count>LIMITS.parts)throw new Error("Recorded equipment outline bound exceeded");
        const matrix=object.shape==="compound"?poseMatrix(math,part.position,part.quaternion):math.identity();
        for(const path of shapePaths(part)){vertices+=path.length;paths.push(path.map(p=>math.point(matrix,p)));}
      }
      if(vertices>LIMITS.vertices)throw new Error("Recorded equipment vertex bound exceeded");
      objects.set(object.id,{paths,color:/^#[0-9a-f]{6}$/i.test(object.color||"")?object.color:"#965644"});
    }
    const floor=math.prepareWorld({...config,objects:[]}),span=Math.max(...config.bounds);
    const floorPaths=[];
    for(let i=0;i<=4;i++){
      const v=(i/2-1)*span;
      floorPaths.push([[-span,v,0],[span,v,0]],[[v,-span,0],[v,span,0]]);
    }
    return {objects,floor:floorPaths.map(path=>path.map(p=>math.point(floor.floor_matrix,p))),vertices};
  }
  function fallbackFrame(math,config,outlines,snapshot,camera,showMuscles) {
    const state=unwrap(snapshot),human=state?.human;
    if(!human||!Array.isArray(human.segments)||human.segments.length>128||!Array.isArray(state.objects)||state.objects.length!==outlines.objects.size)throw new Error("Incomplete original recorded body/world state");
    const paths=outlines.floor.map(points=>({points,color:"#cdbba8",width:1})),seen=new Set();
    for(const object of state.objects){
      const shape=outlines.objects.get(object.id);
      if(!shape||seen.has(object.id))throw new Error("Recorded equipment identity differs");seen.add(object.id);
      const matrix=poseMatrix(math,object.position,object.quaternion);
      for(const points of shape.paths)paths.push({points:points.map(p=>math.point(matrix,p)),color:shape.color,width:1.5});
    }
    const vector=p=>{if(!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isFinite(v)||Math.abs(v)>1e6))throw new Error("Invalid recorded segment");return p;};
    for(const segment of human.segments)paths.push({points:[vector(segment.a),vector(segment.b)],color:"#214F50",width:4});
    if(showMuscles){
      if(!Array.isArray(human.muscles)||human.muscles.length>128)throw new Error("Invalid recorded muscle illustration");
      for(const muscle of human.muscles)paths.push({points:[vector(muscle.a),vector(muscle.b)],color:"#C7795D",width:2});
    }
    return math.cameraViews(config,snapshot,camera,1080,420).map(view=>({name:view.name,viewport:view.viewport,
      paths:paths.map(path=>({color:path.color,width:path.width,points:path.points.map(p=>{const q=math.point(view.matrix,p);return [view.viewport[0]+(q[0]+1)*view.viewport[2]/2,(1-q[1])*view.viewport[3]/2];})}))}));
  }
  function decode(value,maximum) {
    if(typeof value!=="string"||value.length>Math.ceil(maximum/3)*4)throw new Error("Embedded avatar bytes exceed their bound");
    const raw=root.atob(value);if(raw.length>maximum)throw new Error("Embedded avatar bytes exceed their bound");
    const bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return bytes.buffer;
  }
  function installDataWorker() {
    let data=null,scene=null,config=null,outlines=null,fallbackError=null;
    root.onmessage=event=>{
      const request=event.data;
      try{
        if(request.type==="init"){
          if(typeof request.raw!=="string"||request.raw.length>256*1024*1024)throw new Error("Offline recording exceeds its parse bound");
          data=JSON.parse(request.raw);
          if(data.profile!=="weft-offline-avatar-replay-v1"||!Array.isArray(data.events)||!data.events.length||data.events.length>LIMITS.events||!Array.isArray(data.changes)||!data.changes.length||data.changes.length>LIMITS.changes)throw new Error("Unsupported offline recording");
          scene=compactScene(data.scene);
          try{config=root.WeftAvatarRetarget.compileScene(scene);outlines=prepareOutlines(root.WeftAvatarRetarget,config);}catch(error){fallbackError=error.message;}
          root.postMessage({type:"ready",count:data.events.length,scene,resets:data.events.filter(x=>x.type==="reset").map(x=>x.index),
            workerSource:new TextDecoder().decode(decode(data.worker_base64,512*1024)),
            assets:Object.fromEntries(Object.entries(data.assets).map(([key,a])=>[key,{display_name:a.display_name,render_sha256:a.render_sha256}]))});
        }else if(request.type==="select"){
          if(!data||!Number.isInteger(request.index)||request.index<0||request.index>=data.events.length)throw new Error("Invalid physical event index");
          const physical=data.events[request.index],change=settingsAt(data.changes,request.index),settings=change.settings;
          let fallback=null,error=fallbackError;
          try{if(outlines)fallback=fallbackFrame(root.WeftAvatarRetarget,config,outlines,physical.snapshot,request.camera||settings.camera,settings.show_muscles);}catch(caught){error=caught.message;}
          const asset=data.assets[settings.asset],transfer=[];let packageBytes=null;
          if(settings.asset&&asset&&settings.asset!==request.loadedAsset){
            const renderBytes=decode(asset.render_base64,8*1024*1024),textures=asset.textures_base64.map(raw=>decode(raw,32*1024*1024));
            if(textures.reduce((n,b)=>n+b.byteLength,0)>32*1024*1024)throw new Error("Embedded texture aggregate exceeds its bound");
            packageBytes={renderBytes,renderSha256:asset.render_sha256,textures};transfer.push(renderBytes,...textures);
          }
          root.postMessage({type:"selected",requestId:request.requestId,index:request.index,event:physical,change,
            fallback,fallbackError:error,packageBytes,assetMissing:!!settings.asset&&!asset,camera:request.camera||settings.camera,
            inspectionCamera:!!request.camera},transfer);
        }else throw new Error("Unsupported offline replay request");
      }catch(error){root.postMessage({type:"error",requestId:request?.requestId,reason:error.message});}
    };
  }
  function mount(document=root.document) {
    const get=id=>document.getElementById(id),ui=Object.fromEntries(["load-status","previous","play","next","next-reset","scrub","event-label","outcome","controller-label","avatar-canvas","fallback-canvas","appearance-status","appearance-label","inspect-camera","yaw","zoom","follow","recorded-camera","camera-label","event-details","event-json"].map(id=>[id,get(id)]));
    let worker=null,viewer=null,viewerVisible=null,header=null,selected=null,index=0,playing=false,timer=null,inFlight=false,pending=null,sequence=0,disposed=false,loadedAsset=null,avatarStatus="idle",avatarReason="",workerURL=null;
    const urls=[];
    const url=source=>{const value=root.URL.createObjectURL(new Blob([source],{type:"application/javascript"}));urls.push(value);return value;};
    const status=text=>{ui["load-status"].textContent=text;};
    const pause=()=>{playing=false;root.clearTimeout(timer);timer=null;ui.play.textContent="Play";ui.play.setAttribute("aria-pressed","false");};
    function cameraOverride(){return ui["inspect-camera"].checked?{yaw:Number(ui.yaw.value),zoom:Number(ui.zoom.value),follow:ui.follow.checked}:null;}
    function details(){if(selected&&ui["event-details"].open)ui["event-json"].textContent=JSON.stringify({event:selected.event,appearance_change:selected.change},null,2);}
    function appearanceStatus(){
      if(!selected)return;
      const settings=selected.change.settings,key=settings.asset,name=header.assets[key]?.display_name||`Missing recorded avatar ${String(key).slice(0,12)}`;
      const ready=!!key&&!selected.assetMissing&&avatarStatus==="ready";
      ui["avatar-canvas"].hidden=!ready;ui["fallback-canvas"].hidden=ready;
      ui["appearance-label"].textContent=`Recorded appearance change ${selected.change.index} after physical event ${selected.change.after_event_index} · ${key?name:"Original body"} · skeleton overlay ${settings.overlay?"on":"off"} · illustrative muscles ${settings.show_muscles?"on":"off"}`;
      ui["appearance-status"].textContent=ready?`${name} · recorded appearance only`:selected.fallbackError?`Original-body view unavailable: ${selected.fallbackError}`:!key?"Original recorded body: no avatar selected":selected.assetMissing?`${name} is unavailable; showing original recorded body`:`${avatarStatus==="loading"?"Loading "+name:avatarReason||"Avatar rendering unavailable"}; showing original recorded body`;
    }
    const syncVisibility=(force=false)=>{const visible=!!selected?.change.settings.asset&&!selected?.assetMissing&&avatarStatus==="ready"&&!document.hidden;if(viewer&&(force||visible!==viewerVisible)){viewerVisible=visible;viewer.setVisible(visible);}};
    function drawFallback(views){
      const context=ui["fallback-canvas"].getContext("2d");if(!context)return;
      context.clearRect(0,0,1080,420);context.fillStyle="#fff9f2";context.fillRect(0,0,1080,420);
      if(!views)return;
      for(const view of views){
        const [x,y,w,h]=view.viewport;context.save();context.beginPath();context.rect(x,y,w,h);context.clip();
        for(const path of view.paths){context.beginPath();context.strokeStyle=path.color;context.lineWidth=path.width;path.points.forEach((p,i)=>i?context.lineTo(...p):context.moveTo(...p));context.stroke();}
        context.fillStyle="#214F50";context.font="14px system-ui";context.fillText(view.name,x+10,20);context.restore();
      }
    }
    function updateSelected(message){
      selected=message;index=message.index;ui.scrub.value=String(index);ui.scrub.setAttribute("aria-valuetext",`Physical event ${index} of ${header.count-1}: ${describe(message.event).type}`);
      const label=describe(message.event);ui["event-label"].textContent=`Physical event ${index} / ${header.count-1} · ${label.type}`;ui.outcome.textContent=label.outcome;ui["controller-label"].textContent=label.controller;
      ui.previous.disabled=index===0;ui.next.disabled=index===header.count-1;ui["next-reset"].disabled=!header.resets.some(value=>value>index);
      const settings=message.change.settings;
      if(!ui["inspect-camera"].checked){ui.yaw.value=String(settings.camera.yaw);ui.zoom.value=String(settings.camera.zoom);ui.follow.checked=settings.camera.follow;}
      ui["camera-label"].textContent=message.inspectionCamera?"Inspection camera override — recorded camera and physical events are unchanged":"Recorded camera — follows saved appearance changes";
      drawFallback(message.fallback);
      if(settings.asset&&!message.assetMissing){
        if(!viewer){
          try{viewer=root.WeftAvatarViewer.create(ui["avatar-canvas"],{workerURL,onStatus:message=>{const changed=avatarStatus!==message.status;avatarStatus=message.status;avatarReason=message.reason||"";appearanceStatus();if(changed)syncVisibility();}});viewer.configure(header.scene);}catch(error){avatarStatus="fallback";avatarReason=error.message;}
        }
        if(message.packageBytes){loadedAsset=settings.asset;avatarStatus="loading";viewer?.load(message.packageBytes).then(result=>{if(!disposed&&loadedAsset===settings.asset&&result.status!=="cancelled"){avatarStatus=result.status;avatarReason=result.reason||"";appearanceStatus();}});}
        viewer?.update({snapshot:message.event.snapshot,camera:message.camera,overlay:settings.overlay,showMuscles:settings.show_muscles});
      }else loadedAsset=null;
      appearanceStatus();syncVisibility();details();status(`Ready · ${header.count} original physical events retained · saved states only`);
    }
    function drain(){
      if(!worker||!header||inFlight||!pending||disposed||document.hidden)return;
      const request=pending;pending=null;inFlight=true;worker.postMessage({type:"select",...request,loadedAsset});
    }
    function request(next){if(!header||disposed)return;index=Math.max(0,Math.min(header.count-1,next));pending={index,requestId:++sequence,camera:cameraOverride()};drain();}
    function schedule(){
      root.clearTimeout(timer);timer=null;if(!playing||disposed||document.hidden||inFlight||pending)return;
      if(index>=header.count-1){pause();return;}
      const dt=Number(header.scene.dt);timer=root.setTimeout(()=>{timer=null;request(index+1);},Number.isFinite(dt)&&dt>0?Math.min(1000,Math.max(16,dt*1000)):1000/30);
    }
    function playPause(){if(playing){pause();return;}if(!header||document.hidden||disposed)return;playing=true;ui.play.textContent="Pause";ui.play.setAttribute("aria-pressed","true");if(index===header.count-1)request(0);else schedule();}
    function inspection(){const enabled=ui["inspect-camera"].checked;for(const key of ["yaw","zoom","follow","recorded-camera"])ui[key].disabled=!enabled;request(index);}
    function dispose(){if(disposed)return;disposed=true;pause();worker?.terminate();worker=null;viewer?.dispose();for(const value of urls)root.URL.revokeObjectURL(value);}
    ui.play.addEventListener("click",playPause);ui.previous.addEventListener("click",()=>{pause();request(index-1);});ui.next.addEventListener("click",()=>{pause();request(index+1);});
    ui["next-reset"].addEventListener("click",()=>{pause();const next=header?.resets.find(value=>value>index);if(next!==undefined)request(next);});
    ui.scrub.addEventListener("input",()=>{pause();request(Number(ui.scrub.value));});ui["inspect-camera"].addEventListener("change",inspection);
    for(const key of ["yaw","zoom","follow"])ui[key].addEventListener(key==="follow"?"change":"input",()=>request(index));
    ui["recorded-camera"].addEventListener("click",()=>{ui["inspect-camera"].checked=false;inspection();});ui["event-details"].addEventListener("toggle",details);
    document.addEventListener("keydown",event=>{if(!header||disposed||["INPUT","BUTTON","SELECT","TEXTAREA"].includes(event.target?.tagName))return;if(event.key===" "){event.preventDefault();playPause();}else if(["ArrowLeft","ArrowRight","Home","End"].includes(event.key)){event.preventDefault();pause();request(event.key==="Home"?0:event.key==="End"?header.count-1:index+(event.key==="ArrowLeft"?-1:1));}});
    document.addEventListener("visibilitychange",()=>{if(document.hidden){pause();syncVisibility(true);}else{request(index);appearanceStatus();Promise.resolve().then(()=>syncVisibility(true));}});
    root.addEventListener?.("pagehide",event=>{pause();if(!event?.persisted)dispose();else viewer?.setVisible(false);});
    root.addEventListener?.("pageshow",()=>{if(!disposed){request(index);Promise.resolve().then(()=>syncVisibility(true));}});
    try{
      if(typeof root.Worker!=="function")throw new Error("This browser needs worker support to parse the recording safely");
      const math=get("avatar-replay-math").textContent,controller=get("avatar-replay-controller").textContent;
      worker=new root.Worker(url(blockedImports+math+"\n;\n"+controller));
      worker.onmessage=event=>{
        const message=event.data;if(disposed)return;
        if(message.type==="ready"){
          header=message;workerURL=url(blockedImports+math+"\n;\n"+message.workerSource);
          ui.scrub.max=String(header.count-1);ui.scrub.disabled=false;ui.play.disabled=false;request(0);
        }else if(message.type==="selected"){
          inFlight=false;if(message.requestId===sequence&&!document.hidden)updateSelected(message);drain();schedule();
        }else if(message.type==="error"){inFlight=false;pause();status(`Replay unavailable: ${message.reason}`);drain();}
      };
      worker.onerror=event=>{pause();status(`Replay worker failed: ${event.message||"unknown error"}`);worker?.terminate();worker=null;};
      worker.postMessage({type:"init",raw:get("avatar-replay-data").textContent});
    }catch(error){pause();status(`Replay unavailable: ${error.message}`);}
    return Object.freeze({dispose,request,pause,selected:()=>selected});
  }
  const api=Object.freeze({LIMITS,settingsAt,compactScene,describe,prepareOutlines,fallbackFrame,mount});
  root.WeftAvatarReplay=api;
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root.document?.getElementById("avatar-replay-data"))mount();
  else if(!root.document&&typeof root.postMessage==="function")installDataWorker();
})(globalThis);
