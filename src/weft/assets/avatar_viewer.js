"use strict";
((root) => {
  const MAX_RENDER_BYTES=8*1024*1024,MAX_TEXTURE_BYTES=32*1024*1024;
  function compactScene(scene){
    const h=scene?.human_spec;
    if(!Array.isArray(scene?.objects)||scene.objects.length>12||scene.objects.some(object=>{const parts=object.geometry?.parts??object.parts;return parts&&(!Array.isArray(parts)||parts.length>32);}))throw new Error("Avatar world exceeds supported bounds");
    return {profile:scene?.profile,bounds:scene?.bounds,coordinate_frame:{scale_bridge:scene?.coordinate_frame?.scale_bridge},
      human_spec:{lengths:h?.lengths,body_frame:h?.body_frame,mobility:{base_height:h?.mobility?.base_height,ground_plane:h?.mobility?.ground_plane}},
      objects:scene?.objects?.map(object=>({...Object.fromEntries(["id","shape","role","size","radius","length","color","position","quaternion"].filter(key=>object[key]!==undefined).map(key=>[key,object[key]])),...(object.shape==="compound"?{parts:object.geometry?.parts??object.parts}:{})}))};
  }
  function compactSnapshot(snapshot,showMuscles){
    const state=snapshot?.profile==="shared-exercise-state-v1"?snapshot.engine_state:snapshot,h=state?.human;
    if(!h||!Array.isArray(state.objects)||state.objects.length>12||Object.keys(h.angles||{}).length>32||Object.keys(h.joints||{}).length>32||(h.segments||[]).length>32||(h.muscles||[]).length>64)throw new Error("Avatar has no bounded exact world state");
    const vector=value=>{if(!Array.isArray(value)||value.length>4)throw new Error("Invalid avatar state vector");return value.slice();};
    return {human:{angles:{...h.angles},mobility:{translation:vector(h.mobility?.translation)},
      joints:Object.fromEntries(Object.entries(h.joints||{}).map(([key,value])=>[key,vector(value)])),
      segments:(h.segments||[]).map(segment=>({a:vector(segment.a),b:vector(segment.b)})),
      muscles:showMuscles?(h.muscles||[]).map(m=>({a:vector(m.a),b:vector(m.b),activation:m.activation})):[]},
      objects:state.objects.map(object=>({id:object.id,role:object.role,position:vector(object.position),quaternion:vector(object.quaternion)}))};
  }
  class Viewer {
    constructor(canvas,{workerURL="/avatar_worker.js",onStatus=()=>{}}={}){
      this.canvas=canvas;this.onStatus=onStatus;this.worker=null;this.disposed=false;this.ready=false;this.panelVisible=true;this.pageActive=true;this.visible=!root.document?.hidden;
      this.generation=0;this.sequence=0;this.inFlight=null;this.pendingFrame=null;this.lastFrame=null;this.loading=null;this.pendingLoad=null;this.loadTimer=null;this.frameTimer=null;this.disposeTimer=null;
      this.hidden=()=>this.updateVisibility();
      this.pageHide=()=>{this.pageActive=false;this.updateVisibility();};this.pageShow=()=>{this.pageActive=true;this.updateVisibility();};
      try{
        if(typeof root.Worker!=="function"||typeof canvas?.transferControlToOffscreen!=="function")throw new Error("Worker WebGL2 is unavailable; showing the original body");
        const worker=new root.Worker(workerURL);this.worker=worker;
        worker.onmessage=event=>this.receive(event.data);
        worker.onerror=event=>this.fallback(event.message||"Avatar worker failed");
        const offscreen=canvas.transferControlToOffscreen();
        worker.postMessage({type:"init",canvas:offscreen,width:canvas.width,height:canvas.height},[offscreen]);
        root.document?.addEventListener("visibilitychange",this.hidden);
        root.addEventListener?.("pagehide",this.pageHide);root.addEventListener?.("pageshow",this.pageShow);
        this.status("idle","Choose an imported rigged avatar");
      }catch(error){this.fallback(error.message);}
    }
    status(status,reason,extra={}){const signature=`${status}:${reason}`;if(signature===this.lastStatus)return;this.lastStatus=signature;try{this.onStatus({status,reason,...extra});}catch(_){}}
    configure(scene){
      if(this.disposed||!this.worker)return false;
      try{this.worker.postMessage({type:"configure",scene:compactScene(scene)});return true;}catch(error){this.fallback(error.message);return false;}
    }
    load({renderBytes,renderSha256,textures=[]}){
      if(this.disposed||!this.worker)return Promise.resolve({status:"fallback",reason:"Avatar worker is unavailable"});
      if(!(renderBytes instanceof ArrayBuffer)||renderBytes.byteLength===0||renderBytes.byteLength>MAX_RENDER_BYTES||!/^[0-9a-f]{64}$/.test(renderSha256)||!Array.isArray(textures)||textures.length>8||textures.some(bytes=>!(bytes instanceof ArrayBuffer))||textures.reduce((sum,bytes)=>sum+bytes.byteLength,0)>MAX_TEXTURE_BYTES){
        this.ready=false;this.generation++;this.pendingLoad?.resolve({status:"cancelled"});this.pendingLoad=null;if(this.loading)this.worker.postMessage({type:"cancelLoad",generation:this.generation});
        this.status("fallback","Avatar byte package exceeds supported bounds");return Promise.resolve({status:"fallback",reason:"Invalid avatar byte package"});
      }
      const generation=++this.generation;
      this.ready=false;this.status("loading","Loading a verified appearance; physics is unchanged");
      return new Promise(resolve=>{
        if(this.pendingLoad)this.pendingLoad.resolve({status:"cancelled"});
        this.pendingLoad={generation,renderBytes,renderSha256,textures,resolve};
        if(this.loading)this.worker.postMessage({type:"cancelLoad",generation});
        this.drainLoad();
      });
    }
    drainLoad(){
      if(!this.worker||this.loading||!this.pendingLoad||this.disposed)return;
      this.loading=this.pendingLoad;this.pendingLoad=null;
      const {generation,renderBytes,renderSha256,textures}=this.loading;
      this.worker.postMessage({type:"load",generation,renderBytes,renderSha256,textures},[renderBytes,...textures]);
      this.loadTimer=root.setTimeout(()=>this.fallback("Avatar loading exceeded 15 seconds; showing the original body"),15000);
    }
    update({snapshot,camera={},overlay=false,showMuscles=false}){
      if(this.disposed)return;
      try{
        const frame={type:"frame",sequence:++this.sequence,snapshot:compactSnapshot(snapshot,showMuscles),camera:{yaw:camera.yaw??.5,zoom:camera.zoom??1,follow:camera.follow===true},overlay:overlay===true,showMuscles:showMuscles===true};
        this.lastFrame=frame;this.pendingFrame=frame;this.drainFrame();
      }catch(error){this.status("fallback",error.message);}
    }
    drainFrame(){
      if(!this.worker||!this.ready||!this.visible||this.loading||this.pendingLoad||this.inFlight!==null||!this.pendingFrame||this.disposed)return;
      const frame=this.pendingFrame;this.pendingFrame=null;this.inFlight=frame.sequence;
      this.worker.postMessage(frame);
      this.frameTimer=root.setTimeout(()=>this.fallback("Avatar frame exceeded two seconds; showing the original body"),2000);
    }
    setVisible(visible){
      if(this.disposed)return;
      this.panelVisible=visible===true;this.updateVisibility();
    }
    updateVisibility(){
      if(this.disposed)return;
      const visible=this.panelVisible&&this.pageActive&&!root.document?.hidden;
      if(visible===this.visible)return;
      this.visible=visible;this.worker?.postMessage({type:"visible",visible:this.visible});
      if(this.visible){this.pendingFrame=this.lastFrame;this.drainFrame();}
    }
    receive(message){
      if(message?.type==="disposed"){this.finishDispose();return;}
      if(this.disposed)return;
      if(message?.type==="fatal"){this.fallback(message.reason);return;}
      if(message?.type==="loaded"||message?.type==="loadError"||message?.type==="loadCancelled"){
        if(message.generation!==this.loading?.generation)return;
        root.clearTimeout(this.loadTimer);this.loadTimer=null;
        const load=this.loading;this.loading=null;
        const current=message.generation===this.generation&&!this.pendingLoad;
        this.ready=current&&message.type==="loaded";
        load.resolve({status:!current?"cancelled":this.ready?"ready":"fallback",reason:message.reason,stats:message.stats});
        if(current)this.status(this.ready?"ready":"fallback",this.ready?"Rigged avatar; appearance only":message.reason||"Avatar is unavailable",{stats:message.stats});
        this.drainLoad();this.drainFrame();return;
      }
      if(message?.type==="frame"&&message.sequence===this.inFlight){
        this.inFlight=null;root.clearTimeout(this.frameTimer);this.frameTimer=null;
        if(message.error){this.ready=false;this.pendingFrame=null;this.status("fallback",message.error);}
        else if(this.ready)this.status("ready","Rigged avatar; appearance only",{frame:message.sequence});
        this.drainFrame();
      }
    }
    fallback(reason){
      this.ready=false;root.clearTimeout(this.loadTimer);root.clearTimeout(this.frameTimer);this.loadTimer=this.frameTimer=null;
      this.loading?.resolve({status:"fallback",reason});this.pendingLoad?.resolve({status:"fallback",reason});
      this.loading=this.pendingLoad=null;this.pendingFrame=null;this.inFlight=null;
      this.worker?.terminate();this.worker=null;this.status("fallback",reason||"Avatar is unavailable; showing the original body");
    }
    finishDispose(){root.clearTimeout(this.disposeTimer);this.disposeTimer=null;this.worker?.terminate();this.worker=null;}
    dispose(){
      if(this.disposed)return;this.disposed=true;this.ready=false;root.clearTimeout(this.loadTimer);root.clearTimeout(this.frameTimer);this.loadTimer=this.frameTimer=null;
      this.loading?.resolve({status:"cancelled"});this.pendingLoad?.resolve({status:"cancelled"});this.loading=this.pendingLoad=null;this.pendingFrame=this.lastFrame=null;
      root.document?.removeEventListener("visibilitychange",this.hidden);root.removeEventListener?.("pagehide",this.pageHide);root.removeEventListener?.("pageshow",this.pageShow);
      if(this.worker){this.worker.postMessage({type:"dispose"});this.disposeTimer=root.setTimeout(()=>this.finishDispose(),100);}
    }
  }
  const api=Object.freeze({create:(canvas,options)=>new Viewer(canvas,options)});
  root.WeftAvatarViewer=api;if(typeof module!=="undefined"&&module.exports)module.exports=api;
})(globalThis);
