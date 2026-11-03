"use strict";
if(!self.WeftAvatarRetarget)importScripts("/avatar_retarget.js");
(() => {
  const math=self.WeftAvatarRetarget;
  let canvas=null,gl=null,program=null,lineProgram=null,paletteTexture=null,whiteTexture=null,lineVao=null,linePosition=null,lineColor=null;
  let config=null,world=null,worldMeshes=[],asset=null,assetMeshes=[],assetTextures=[],visible=true,closed=false,requestedGeneration=0;
  const VERTEX=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUv;
layout(location=3) in vec4 aJoints;
layout(location=4) in vec4 aWeights;
uniform mat4 uView;
uniform mat4 uNode;
uniform bool uSkinned;
uniform sampler2D uPalette;
out vec3 vNormal;
out vec2 vUv;
mat4 bone(int index){return mat4(texelFetch(uPalette,ivec2(0,index),0),texelFetch(uPalette,ivec2(1,index),0),texelFetch(uPalette,ivec2(2,index),0),texelFetch(uPalette,ivec2(3,index),0));}
void main(){
  mat4 skin=uSkinned?bone(int(aJoints.x))*aWeights.x+bone(int(aJoints.y))*aWeights.y+bone(int(aJoints.z))*aWeights.z+bone(int(aJoints.w))*aWeights.w:uNode;
  mat3 basis=mat3(skin);
  vNormal=abs(determinant(basis))>1e-12?transpose(inverse(basis))*aNormal:aNormal;
  vUv=aUv;gl_Position=uView*skin*vec4(aPosition,1.0);
}`;
  const FRAGMENT=`#version 300 es
precision highp float;
in vec3 vNormal;
in vec2 vUv;
uniform sampler2D uTexture;
uniform vec4 uColor;
uniform bool uUnlit;
uniform bool uMask;
uniform bool uDoubleSided;
uniform float uCutoff;
out vec4 color;
vec3 encodeSRGB(vec3 c){c=max(c,vec3(0.0));return mix(1.055*pow(c,vec3(1.0/2.4))-.055,12.92*c,lessThanEqual(c,vec3(.0031308)));}
void main(){
  vec4 base=texture(uTexture,vUv)*uColor;if(uMask&&base.a<uCutoff)discard;
  vec3 n=length(vNormal)>1e-12?normalize(vNormal):vec3(0.0,0.0,1.0);if(uDoubleSided&&!gl_FrontFacing)n=-n;
  float light=uUnlit?1.0:.35+.65*max(dot(n,normalize(vec3(.4,-.5,.8))),0.0);
  color=vec4(encodeSRGB(base.rgb*light),1.0);
}`;
  const LINE_VERTEX=`#version 300 es
precision highp float;layout(location=0) in vec3 aPosition;layout(location=1) in vec3 aColor;uniform mat4 uView;out vec3 vColor;
void main(){vColor=aColor;gl_Position=uView*vec4(aPosition,1.0);gl_PointSize=3.0;}`;
  const LINE_FRAGMENT=`#version 300 es
precision highp float;in vec3 vColor;out vec4 color;void main(){color=vec4(vColor,1.0);}`;
  function send(value){if(!closed)self.postMessage(value);}
  function shader(type,source){const item=gl.createShader(type);gl.shaderSource(item,source);gl.compileShader(item);if(!gl.getShaderParameter(item,gl.COMPILE_STATUS)){const reason=gl.getShaderInfoLog(item);gl.deleteShader(item);throw new Error(`Avatar shader compilation failed: ${reason}`);}return item;}
  function buildProgram(vertex,fragment,names){
    const v=shader(gl.VERTEX_SHADER,vertex),f=shader(gl.FRAGMENT_SHADER,fragment),handle=gl.createProgram();
    gl.attachShader(handle,v);gl.attachShader(handle,f);gl.linkProgram(handle);gl.deleteShader(v);gl.deleteShader(f);
    if(!gl.getProgramParameter(handle,gl.LINK_STATUS)){const reason=gl.getProgramInfoLog(handle);gl.deleteProgram(handle);throw new Error(`Avatar shader link failed: ${reason}`);}
    return {handle,uniforms:Object.fromEntries(names.map(name=>[name,gl.getUniformLocation(handle,name)]))};
  }
  function createMesh(primitive){
    const vao=gl.createVertexArray(),buffers=[];gl.bindVertexArray(vao);
    for(const [location,data,size,type] of [[0,primitive.positions,3,gl.FLOAT],[1,primitive.normals,3,gl.FLOAT],[2,primitive.texcoords,2,gl.FLOAT],[3,primitive.joints,4,gl.UNSIGNED_SHORT],[4,primitive.weights,4,gl.FLOAT]]){
      if(data){const buffer=gl.createBuffer();buffers.push(buffer);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,type,false,0,0);}
      else{gl.disableVertexAttribArray(location);if(location===2)gl.vertexAttrib2f(location,0,0);else if(location===3||location===4)gl.vertexAttrib4f(location,0,0,0,0);}
    }
    const index=gl.createBuffer();buffers.push(index);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,index);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,primitive.indices,gl.STATIC_DRAW);gl.bindVertexArray(null);
    return {vao,buffers,count:primitive.indices.length,primitive};
  }
  function releaseMeshes(meshes){if(gl)for(const mesh of meshes){gl.deleteVertexArray(mesh.vao);for(const buffer of mesh.buffers)gl.deleteBuffer(buffer);}}
  function releaseAsset(){releaseMeshes(assetMeshes);assetMeshes=[];if(gl)for(const texture of assetTextures)gl.deleteTexture(texture);assetTextures=[];asset=null;}
  function initialize(message){
    canvas=message.canvas;
    if(!Number.isInteger(message.width)||!Number.isInteger(message.height)||message.width<3||message.height<1||message.width>2048||message.height>2048)throw new Error("Avatar canvas exceeds supported bounds");
    canvas.width=message.width;canvas.height=message.height;
    gl=canvas.getContext("webgl2",{alpha:false,antialias:true,preserveDrawingBuffer:false});
    if(!gl||gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS)<1||gl.getParameter(gl.MAX_TEXTURE_SIZE)<2048)throw new Error("Worker WebGL2 skinning is unavailable; showing the original body");
    canvas.addEventListener?.("webglcontextlost",event=>{event.preventDefault();send({type:"fatal",reason:"Avatar graphics context was lost; showing the original body"});});
    program=buildProgram(VERTEX,FRAGMENT,["uView","uNode","uSkinned","uPalette","uTexture","uColor","uUnlit","uMask","uCutoff","uDoubleSided"]);
    lineProgram=buildProgram(LINE_VERTEX,LINE_FRAGMENT,["uView"]);
    paletteTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,paletteTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,4,64,0,gl.RGBA,gl.FLOAT,new Float32Array(4*64*4));
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    whiteTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,whiteTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.SRGB8_ALPHA8,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([255,255,255,255]));gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    lineVao=gl.createVertexArray();gl.bindVertexArray(lineVao);
    linePosition=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,linePosition);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
    lineColor=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,lineColor);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,0,0);gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.BLEND);
  }
  const sha=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),value=>value.toString(16).padStart(2,"0")).join("");
  async function load(message){
    const generation=message.generation;requestedGeneration=Math.max(requestedGeneration,generation);
    const textures=[];let meshes=[];
    const cancelled=()=>closed||generation!==requestedGeneration;
    try{
      if(!gl||!config)throw new Error("Avatar world is not configured");
      if(!(message.renderBytes instanceof ArrayBuffer)||message.renderBytes.byteLength>8*1024*1024||!Array.isArray(message.textures)||message.textures.length>8||message.textures.some(bytes=>!(bytes instanceof ArrayBuffer))||message.textures.reduce((sum,bytes)=>sum+bytes.byteLength,0)>32*1024*1024)throw new Error("Avatar byte bounds exceeded");
      if(await sha(message.renderBytes)!==message.renderSha256)throw new Error("Avatar render bytes changed");
      if(cancelled())throw new Error("cancelled");
      const prepared=math.prepareAsset(JSON.parse(new TextDecoder("utf-8",{fatal:true}).decode(message.renderBytes)));
      if(message.textures.length!==prepared.textures.length)throw new Error("Avatar texture set is incomplete");
      releaseAsset();
      for(let index=0;index<prepared.textures.length;index++){
        const info=prepared.textures[index],bytes=message.textures[index];
        if(await sha(bytes)!==info.sha256)throw new Error("Avatar texture bytes changed");
        if(cancelled())throw new Error("cancelled");
        const bitmap=await createImageBitmap(new Blob([bytes],{type:info.mime_type}),{imageOrientation:"none",premultiplyAlpha:"none",colorSpaceConversion:"none"});
        try{
          if(cancelled())throw new Error("cancelled");
          if(bitmap.width!==info.width||bitmap.height!==info.height)throw new Error("Decoded avatar texture dimensions changed");
          const texture=gl.createTexture();textures.push(texture);gl.bindTexture(gl.TEXTURE_2D,texture);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false);gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,gl.NONE);
          gl.texImage2D(gl.TEXTURE_2D,0,gl.SRGB8_ALPHA8,gl.RGBA,gl.UNSIGNED_BYTE,bitmap);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,info.wrap_s);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,info.wrap_t);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,info.mag_filter);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,info.min_filter);
          if(info.min_filter>=9984)gl.generateMipmap(gl.TEXTURE_2D);
        }finally{bitmap.close();}
      }
      if(cancelled())throw new Error("cancelled");
      meshes=prepared.primitives.map(createMesh);
      if(gl.getError()!==gl.NO_ERROR)throw new Error("Avatar graphics allocation failed");
      asset=prepared;assetMeshes=meshes;assetTextures=textures;
      send({type:"loaded",generation,stats:{...prepared.stats,primitives:meshes.length,textures:textures.length}});
    }catch(error){
      releaseMeshes(meshes);if(gl)for(const texture of textures)gl.deleteTexture(texture);
      send({type:cancelled()?"loadCancelled":"loadError",generation,reason:cancelled()?"Avatar load superseded":error.message});
    }
  }
  function material(value){
    const u=program.uniforms;gl.uniform4fv(u.uColor,value.base_color_factor);gl.uniform1i(u.uUnlit,value.unlit);gl.uniform1i(u.uMask,value.alpha_mode==="MASK");gl.uniform1f(u.uCutoff,value.alpha_cutoff);gl.uniform1i(u.uDoubleSided,value.double_sided);
    if(value.double_sided)gl.disable(gl.CULL_FACE);else{gl.enable(gl.CULL_FACE);gl.cullFace(gl.BACK);}
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,value.base_color_texture===null?whiteTexture:assetTextures[value.base_color_texture]);
  }
  function drawMesh(mesh,node,skinned){
    gl.uniformMatrix4fv(program.uniforms.uNode,false,node);gl.uniform1i(program.uniforms.uSkinned,skinned);gl.bindVertexArray(mesh.vao);gl.drawElements(gl.TRIANGLES,mesh.count,gl.UNSIGNED_INT,0);
  }
  function overlayData(state,showSkeleton,showMuscles){
    const positions=[],colors=[];
    const add=(a,b,color)=>{if(!Array.isArray(a)||!Array.isArray(b)||a.length!==3||b.length!==3||[...a,...b].some(value=>!Number.isFinite(value)))throw new Error("Invalid avatar overlay");positions.push(...a,...b);colors.push(...color,...color);};
    if(showSkeleton){const segments=state.human.segments||[];if(segments.length>32)throw new Error("Skeleton overlay exceeds bounds");for(const segment of segments)add(segment.a,segment.b,[33/255,79/255,80/255]);}
    if(showMuscles){const muscles=state.human.muscles||[];if(muscles.length>64)throw new Error("Muscle overlay exceeds bounds");for(const muscle of muscles){const a=muscle.activation;if(!Number.isFinite(a)||a<0||a>1)throw new Error("Invalid illustrative drive");add(muscle.a,muscle.b,[(150+55*a)/255,(92-30*a)/255,(72-20*a)/255]);}}
    return {positions:new Float32Array(positions),colors:new Float32Array(colors)};
  }
  function frame(message){
    try{
      if(!visible||!asset||!world){send({type:"frame",sequence:message.sequence});return;}
      const body=math.bodyFrames(config,message.snapshot),pose=math.retarget(asset,body),matrices=math.worldMatrices(world,body.state),views=math.cameraViews(config,body.state,message.camera,canvas.width,canvas.height,math.posedBounds(asset,pose,world,matrices,body.state));
      const overlays=overlayData(body.state,message.overlay,message.showMuscles);
      gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,paletteTexture);gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,4,pose.palette.length,gl.RGBA,gl.FLOAT,new Float32Array(pose.palette.flat()));
      gl.disable(gl.SCISSOR_TEST);gl.clearColor(251/255,247/255,241/255,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      if(overlays.positions.length){gl.bindBuffer(gl.ARRAY_BUFFER,linePosition);gl.bufferData(gl.ARRAY_BUFFER,overlays.positions,gl.DYNAMIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,lineColor);gl.bufferData(gl.ARRAY_BUFFER,overlays.colors,gl.DYNAMIC_DRAW);}
      for(const view of views){
        gl.viewport(...view.viewport);gl.enable(gl.SCISSOR_TEST);gl.scissor(...view.viewport);gl.useProgram(program.handle);gl.enable(gl.DEPTH_TEST);
        gl.uniformMatrix4fv(program.uniforms.uView,false,view.matrix);gl.uniform1i(program.uniforms.uPalette,1);gl.uniform1i(program.uniforms.uTexture,0);
        for(let i=0;i<worldMeshes.length;i++){material({base_color_factor:world.meshes[i].color,base_color_texture:null,unlit:false,alpha_mode:"OPAQUE",alpha_cutoff:.5,double_sided:false});drawMesh(worldMeshes[i],matrices[i],false);}
        for(const mesh of assetMeshes){material(asset.materials[mesh.primitive.material]);drawMesh(mesh,pose.nodes.get(mesh.primitive.node),mesh.primitive.skin!==null);}
        if(overlays.positions.length){gl.disable(gl.DEPTH_TEST);gl.useProgram(lineProgram.handle);gl.uniformMatrix4fv(lineProgram.uniforms.uView,false,view.matrix);gl.bindVertexArray(lineVao);gl.drawArrays(gl.LINES,0,overlays.positions.length/3);if(message.overlay)gl.drawArrays(gl.POINTS,0,overlays.positions.length/3);}
      }
      gl.disable(gl.SCISSOR_TEST);gl.bindVertexArray(null);
      if(gl.getError()!==gl.NO_ERROR)throw new Error("Avatar graphics frame failed");
      send({type:"frame",sequence:message.sequence});
    }catch(error){send({type:"frame",sequence:message.sequence,error:error.message});}
  }
  function dispose(){
    releaseAsset();releaseMeshes(worldMeshes);worldMeshes=[];
    if(gl){for(const texture of [paletteTexture,whiteTexture])if(texture)gl.deleteTexture(texture);for(const p of [program,lineProgram])if(p)gl.deleteProgram(p.handle);if(lineVao)gl.deleteVertexArray(lineVao);if(linePosition)gl.deleteBuffer(linePosition);if(lineColor)gl.deleteBuffer(lineColor);}
    self.postMessage({type:"disposed"});closed=true;self.close();
  }
  self.onmessage=({data})=>{
    if(closed)return;
    try{
      if(data.type==="init")initialize(data);
      else if(data.type==="configure"){config=math.compileScene(data.scene);world=math.prepareWorld(config);releaseMeshes(worldMeshes);worldMeshes=world.meshes.map(createMesh);}
      else if(data.type==="load")void load(data);
      else if(data.type==="cancelLoad")requestedGeneration=Math.max(requestedGeneration,data.generation);
      else if(data.type==="frame")frame(data);
      else if(data.type==="visible")visible=data.visible===true;
      else if(data.type==="dispose")dispose();
    }catch(error){send({type:"fatal",reason:error.message});}
  };
})();
