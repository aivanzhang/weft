"use strict";
((root) => {
  const LIMITS = Object.freeze({nodes:128, joints:64, primitives:16, vertices:20000, triangles:40000, textures:8, pixels:8388608});
  const BODY_FRAMES = Object.freeze(["mobile_root", "torso", ...["left", "right"].flatMap(side => ["shoulder", "elbow", "hip", "knee"].map(joint => `${side}_${joint}_body`))]);
  const fail = message => { throw new Error(message); };
  const finite = (value, label, bound = 1e6) => typeof value === "number" && Number.isFinite(value) && Math.abs(value) <= bound ? value : fail(`Invalid ${label}`);
  const integer = (value, min, max, label) => Number.isInteger(value) && value >= min && value <= max ? value : fail(`Invalid ${label}`);
  const vector = (value, count, label, bound = 1e6) => {
    if (!Array.isArray(value) || value.length !== count) fail(`Invalid ${label}`);
    return value.map(item => finite(item, label, bound));
  };
  const identity = () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
  function multiply(a, b) {
    const out = new Array(16).fill(0);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    return out;
  }
  function quaternion(value, label) {
    const q = vector(value, 4, label, 1.000001);
    if (Math.abs(Math.hypot(...q) - 1) > 1e-6) fail(`Non-unit ${label}`);
    return q;
  }
  function transform(translation = [0,0,0], q = [0,0,0,1], scale = 1) {
    const [x,y,z,w] = q;
    return [(1-2*y*y-2*z*z)*scale, (2*x*y+2*z*w)*scale, (2*x*z-2*y*w)*scale, 0,
      (2*x*y-2*z*w)*scale, (1-2*x*x-2*z*z)*scale, (2*y*z+2*x*w)*scale, 0,
      (2*x*z+2*y*w)*scale, (2*y*z-2*x*w)*scale, (1-2*x*x-2*y*y)*scale, 0, ...translation, 1];
  }
  function axisRotation(axis, angle) {
    const q = axis.map(value => value * Math.sin(angle / 2));
    return transform([0,0,0], [...q, Math.cos(angle / 2)]);
  }
  function point(matrix, xyz) {
    const w=matrix[3]*xyz[0]+matrix[7]*xyz[1]+matrix[11]*xyz[2]+matrix[15];
    if(Math.abs(w)<1e-12)fail("Invalid homogeneous avatar point");
    return [0,1,2].map(row => (matrix[row] * xyz[0] + matrix[4+row] * xyz[1] + matrix[8+row] * xyz[2] + matrix[12+row])/w);
  }
  function normalVector(matrix,n){
    const [a,d,g,b,e,h,c,f,i]=[matrix[0],matrix[1],matrix[2],matrix[4],matrix[5],matrix[6],matrix[8],matrix[9],matrix[10]];
    const det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if(Math.abs(det)<=1e-12)return normalize(Array.from(n));
    return normalize([(e*i-f*h)*n[0]+(f*g-d*i)*n[1]+(d*h-e*g)*n[2],(c*h-b*i)*n[0]+(a*i-c*g)*n[1]+(b*g-a*h)*n[2],(b*f-c*e)*n[0]+(c*d-a*f)*n[1]+(a*e-b*d)*n[2]].map(value=>value/det));
  }
  function inverse(matrix) {
    const rows = Array.from({length:4}, (_, r) => [...Array.from({length:4}, (_, c) => matrix[c*4+r]), ...Array.from({length:4}, (_, c) => Number(c === r))]);
    for (let c = 0; c < 4; c++) {
      let pivot = c;
      for (let r = c+1; r < 4; r++) if (Math.abs(rows[r][c]) > Math.abs(rows[pivot][c])) pivot = r;
      if (Math.abs(rows[pivot][c]) < 1e-12) fail("Singular avatar transform");
      [rows[c], rows[pivot]] = [rows[pivot], rows[c]];
      const divisor = rows[c][c]; rows[c] = rows[c].map(value => value / divisor);
      for (let r = 0; r < 4; r++) if (r !== c) { const amount = rows[r][c]; rows[r] = rows[r].map((value, i) => value - amount * rows[c][i]); }
    }
    return Array.from({length:16}, (_, i) => rows[i%4][4+Math.floor(i/4)]);
  }
  function matrix(value, label) {
    const result = vector(value, 16, label);
    if (Math.abs(result[3])+Math.abs(result[7])+Math.abs(result[11])+Math.abs(result[15]-1) > 1e-6) fail(`Non-affine ${label}`);
    inverse(result);
    return result;
  }
  const dot = (a,b) => a.reduce((sum,x,i) => sum+x*b[i],0);
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const normalize = value => { const n = Math.hypot(...value); if (n < 1e-12) fail("Degenerate avatar axis"); return value.map(x => x/n); };
  function compileScene(scene) {
    if (!["shared-world-v1", "shared-world-v2", "shared-world-v3"].includes(scene?.profile)) fail("Avatar requires a shared-world scene");
    const bridge = scene.coordinate_frame?.scale_bridge, human = scene.human_spec;
    if (!bridge || !human?.mobility || !human.lengths) fail("Missing avatar body frame");
    const scale = finite(bridge.body_height_to_world_scale,"body scale",1000);
    if (scale <= 0) fail("Body scale must be positive");
    const rotation = human.body_frame?.rotation || [[1,0,0],[0,1,0],[0,0,1]];
    if (!Array.isArray(rotation) || rotation.length !== 3) fail("Invalid body orientation");
    const rows = rotation.map(row => vector(row,3,"body orientation",1.000001));
    const x = normalize(rows.map(row => row[0])), rawY = rows.map(row => row[1]);
    const y = normalize(rawY.map((v,i) => v-dot(x,rawY)*x[i])), z = cross(x,y);
    if (Math.max(...rows.flatMap((row,r) => row.map((v,c) => Math.abs(v-[x,y,z][c][r])))) > 1e-6) fail("Invalid body orientation basis");
    const orientation = [...x,0,...y,0,...z,0,0,0,0,1];
    const lengths = {};
    for (const name of ["torso","shoulder_width","hip_width","upper_arm","forearm","thigh","shin"]) {
      lengths[name] = finite(human.lengths[name],`body ${name}`,10)*scale;
      if (lengths[name] <= 0) fail("Body lengths must be positive");
    }
    const objects = scene.objects;
    if (!Array.isArray(objects) || objects.length > 12) fail("Avatar world object bound exceeded");
    const bounds=vector(scene.bounds,3,"world bounds",1000);if(bounds.some(value=>value<=0))fail("Invalid world bounds");
    return {profile:scene.profile, scale, origin:vector(bridge.origin_world,3,"body origin"), base_height:finite(human.mobility.base_height,"body base height",100)*scale,
      orientation, lengths, bounds,
      floor_normal:normalize(vector(human.mobility.ground_plane?.normal || [0,0,1],3,"floor normal",1)),
      objects:objects.map(object => {
        const out = {};
        for (const key of ["id","shape","role","size","radius","length","color","position","quaternion","parts"]) if (object[key] !== undefined) out[key] = object[key];
        if(object.shape==="compound")out.parts=object.geometry?.parts??object.parts;
        if (out.parts && (!Array.isArray(out.parts) || out.parts.length > 32)) fail("Avatar compound bound exceeded");
        return JSON.parse(JSON.stringify(out));
      })};
  }
  function bodyFrames(config, snapshot) {
    const state = snapshot?.profile === "shared-exercise-state-v1" ? snapshot.engine_state : snapshot;
    const human = state?.human;
    if (!human?.angles || !human.mobility || !human.joints) fail("Missing exact human pose");
    const translation = vector(human.mobility.translation,3,"body translation",100);
    const frames = {mobile_root:transform(config.origin.map((value,i) => value+translation[i]*config.scale+(i===2?config.base_height:0)))};
    const angle = name => finite(human.angles[name],`joint ${name}`,100);
    const rotation = (twist,lateral,flex) => multiply(multiply(axisRotation([0,0,1],angle(twist)),axisRotation([0,1,0],angle(lateral))),axisRotation([1,0,0],angle(flex)));
    frames.torso = multiply(multiply(frames.mobile_root,config.orientation),rotation("torso_twist","torso_lateral","torso_flex"));
    const l = config.lengths, expected = {pelvis:point(frames.torso,[0,0,0]),neck:point(frames.torso,[0,0,l.torso]),head:point(frames.torso,[0,0,l.torso+0.18*config.scale])};
    for (const [side,sign] of [["left",-1],["right",1]]) for (const [proximal,distal,tip,width,elevation,length1,length2] of [
      ["shoulder","elbow","wrist",l.shoulder_width,l.torso,l.upper_arm,l.forearm], ["hip","knee","ankle",l.hip_width,0,l.thigh,l.shin]]) {
      const baseName = `${side}_${proximal}`, distalName = `${side}_${distal}`;
      const base = multiply(multiply(frames.torso,transform([sign*width/2,0,elevation])),rotation(`${baseName}_twist`,`${baseName}_lateral`,baseName));
      const child = multiply(multiply(base,transform([0,0,-length1])),axisRotation([distal === "knee"?-1:1,0,0],angle(distalName)));
      frames[`${baseName}_body`] = base; frames[`${distalName}_body`] = child;
      expected[baseName] = point(base,[0,0,0]); expected[distalName] = point(child,[0,0,0]); expected[`${side}_${tip}`] = point(child,[0,0,-length2]);
    }
    const tolerance = 1e-7*Math.max(1,config.scale);
    for (const [name,position] of Object.entries(expected)) {
      const actual = vector(human.joints[name],3,`landmark ${name}`);
      if (Math.hypot(...position.map((value,i) => value-actual[i])) > tolerance) fail(`Avatar FK disagrees with recorded ${name}; use original body`);
    }
    return {frames,landmarks:expected,state};
  }
  function prepareAsset(document) {
    if (document?.profile !== "weft-avatar-render-v1") fail("Unsupported avatar render profile");
    const nodes = document.nodes, skins = document.skins, primitives = document.primitives, materials = document.materials, textures = document.textures;
    if (!Array.isArray(nodes) || !nodes.length || nodes.length > LIMITS.nodes || !Array.isArray(skins) || skins.length !== 1 || !Array.isArray(primitives) || !primitives.length || primitives.length > LIMITS.primitives || !Array.isArray(materials) || !materials.length || materials.length > 32 || !Array.isArray(textures) || textures.length > LIMITS.textures) fail("Avatar structure exceeds supported bounds");
    const byId = new Map();
    for (const node of nodes) {
      const id = integer(node.index,0,LIMITS.nodes-1,"avatar node");
      if (byId.has(id)) fail("Duplicate avatar node");
      byId.set(id,{index:id,parent:node.parent === null ? null : integer(node.parent,0,LIMITS.nodes-1,"avatar parent"),local:matrix(node.local_matrix,"avatar rest matrix")});
    }
    const order = [], visited = new Set(), visiting = new Set();
    function visit(id) {
      if (visited.has(id)) return;
      if (visiting.has(id) || !byId.has(id)) fail("Invalid avatar hierarchy");
      visiting.add(id); const node=byId.get(id); if (node.parent !== null) visit(node.parent);
      visiting.delete(id); visited.add(id); order.push(id);
    }
    for (const id of byId.keys()) visit(id);
    const rig = document.retargeting;
    const restRelative = rig?.profile === "weft-avatar-retarget-v2";
    if (!["weft-avatar-retarget-v1", "weft-avatar-retarget-v2"].includes(rig?.profile) || !Array.isArray(rig.bindings) || !rig.bindings.length || rig.bindings.length > LIMITS.nodes || !byId.has(rig.root_node)) fail("Missing explicit avatar rig mapping");
    if(restRelative&&Object.keys(rig).sort().join(",")!=="bindings,orientation_xyzw,profile,root_node,scale")fail("Unknown rest-relative mapping field");
    const scale = finite(rig.scale,"avatar scale",1000); if (scale <= 0) fail("Avatar scale must be positive");
    const basis = transform([0,0,0],quaternion(rig.orientation_xyzw,"avatar orientation"),scale), bindings = new Map();
    for (const binding of rig.bindings) {
      if (!byId.has(binding.node) || bindings.has(binding.node) || !BODY_FRAMES.includes(binding.body)) fail("Invalid avatar body binding");
      if(restRelative && (Object.keys(binding).sort().join(",")!=="body,node")) fail("Rest-relative bindings only declare node and body");
      bindings.set(binding.node,{body:binding.body,offset:restRelative?null:multiply(transform(vector(binding.translation,3,"avatar offset",100),quaternion(binding.rotation_xyzw,"avatar bone orientation")),basis)});
    }
    if (!bindings.has(rig.root_node)) fail("Avatar root must be explicitly mapped");
    const underRoot = id => { while (id !== null) { if (id === rig.root_node) return true; id=byId.get(id)?.parent ?? null; } return false; };
    for (const id of bindings.keys()) if (!underRoot(id)) fail("Avatar binding lies outside mapped root");
    const skin = skins[0];
    if (!Array.isArray(skin.joints) || !skin.joints.length || skin.joints.length > LIMITS.joints || new Set(skin.joints).size !== skin.joints.length || !Array.isArray(skin.inverse_bind_matrices) || skin.inverse_bind_matrices.length !== skin.joints.length) fail("Invalid avatar skin");
    for (const id of skin.joints) if (!byId.has(id) || !underRoot(id)) fail("Skin joint lies outside mapped root");
    const inverseBind = skin.inverse_bind_matrices.map(value => matrix(value,"inverse bind matrix"));
    const rest=new Map(),restInverse=new Map(),roles=new Map();
    if(restRelative){
      for(const id of order){const node=byId.get(id);rest.set(id,multiply(node.parent===null?basis:rest.get(node.parent),node.local));}
      for(const [id,binding] of bindings){if(roles.has(binding.body))fail("Duplicate rest-relative body role");roles.set(binding.body,id);}
      if(roles.size!==BODY_FRAMES.length||roles.get("mobile_root")!==rig.root_node)fail("Rest-relative avatars require exactly ten body roles");
      for(const [role,id] of roles){
        if(role!=="mobile_root"&&!skin.joints.includes(id))fail("Controlled avatar bone is not skinned");
        const expected=role==="torso"?"mobile_root":role.includes("shoulder")?"torso":role.includes("elbow")?role.replace("elbow","shoulder"):role.includes("knee")?role.replace("knee","hip"):null;
        if(expected){let parent=byId.get(id).parent;while(parent!==null&&!bindings.has(parent))parent=byId.get(parent).parent;if(parent!==roles.get(expected))fail("Incompatible rest-relative limb topology");}
      }
      for(const m of rest.values()){
        if(m.some(v=>!Number.isFinite(v)||Math.abs(v)>1e9))fail("Avatar accumulated rest transform exceeds its numeric bound");
        const cols=[0,4,8].map(i=>m.slice(i,i+3)),lengths=cols.map(v=>Math.hypot(...v));
        if(lengths.some(v=>Math.abs(v-lengths[0])>1e-5*Math.max(1,lengths[0]))||cols.some((a,i)=>cols.some((b,j)=>Math.abs(dot(a,b)/(lengths[i]*lengths[j])-Number(i===j))>1e-5))||dot(cross(cols[0],cols[1]),cols[2])<=0)fail("Rest-relative avatar requires uniform nonsheared rest transforms");
      }
      const bind=skin.joints.map((id,i)=>multiply(rest.get(id),inverseBind[i]));
      if(bind.some(m=>m.some((v,i)=>Math.abs(v-bind[0][i])>2e-5*Math.max(1,Math.abs(bind[0][i])))))fail("Rest-relative avatar default pose disagrees with skin bind pose");
      if(["left","right"].reduce((sum,side)=>sum+rest.get(roles.get(`${side}_shoulder_body`))[14]-rest.get(roles.get(`${side}_hip_body`))[14],0)<=2e-6)fail("Avatar orientation must place shoulders above hips");
      for(const [side,sign] of [["left",-1],["right",1]]){
        for(const joint of ["shoulder","hip"])if(sign*(rest.get(roles.get(`${side}_${joint}_body`))[12]-rest.get(roles.get("torso"))[12])<=1e-6)fail("Avatar left/right mapping contradicts simulation lateral convention");
        for(const [joint,parent] of [["elbow","shoulder"],["knee","hip"]])if(Math.hypot(...[12,13,14].map(i=>rest.get(roles.get(`${side}_${joint}_body`))[i]-rest.get(roles.get(`${side}_${parent}_body`))[i]))<=1e-6)fail("Avatar limb has coincident pivots");
      }
    }
    if(restRelative)for(const [id,m] of rest)restInverse.set(id,inverse(m));
    let pixels=0;
    const textureInfo = textures.map(texture => {
      if (!/^[0-9a-f]{64}$/.test(texture.sha256) || !["image/png","image/jpeg"].includes(texture.mime_type)) fail("Unsupported avatar texture");
      const width=integer(texture.width,1,2048,"texture width"),height=integer(texture.height,1,2048,"texture height"); pixels+=width*height;
      for (const key of ["wrap_s","wrap_t"]) if (![10497,33071,33648].includes(texture[key])) fail("Unsupported texture wrapping");
      if (![9728,9729].includes(texture.mag_filter) || ![9728,9729,9984,9985,9986,9987].includes(texture.min_filter)) fail("Unsupported texture filtering");
      return {...texture,width,height};
    });
    if (pixels > LIMITS.pixels) fail("Avatar decoded texture bound exceeded");
    const materialInfo = materials.map(material => {
      const color=vector(material.base_color_factor,4,"avatar color",1);
      if (color.some(value=>value<0) || typeof material.unlit !== "boolean" || typeof material.double_sided !== "boolean" || !["OPAQUE","MASK"].includes(material.alpha_mode)) fail("Unsupported avatar material");
      const texture=material.base_color_texture;
      if (texture !== null) integer(texture,0,textureInfo.length-1,"avatar material texture");
      const cutoff=finite(material.alpha_cutoff,"alpha cutoff",1);if(cutoff<0)fail("Invalid alpha cutoff");
      return {...material,base_color_factor:color};
    });
    let vertices=0,triangles=0;
    const meshes = primitives.map(primitive => {
      if (!byId.has(primitive.node) || !underRoot(primitive.node) || ![null,0].includes(primitive.skin)) fail("Invalid primitive node or skin");
      if (!Array.isArray(primitive.positions) || primitive.positions.length%3) fail("Invalid avatar positions");
      const count=primitive.positions.length/3;integer(count,3,LIMITS.vertices,"vertex count");vertices+=count;
      if(vertices>LIMITS.vertices)fail("Avatar vertex bound exceeded");
      const positions=new Float32Array(vector(primitive.positions,count*3,"avatar positions"));
      const normals=new Float32Array(vector(primitive.normals,count*3,"avatar normals",1.00001));
      for(let i=0;i<normals.length;i+=3)if(Math.hypot(normals[i],normals[i+1],normals[i+2])<1e-8)fail("Zero avatar normal");
      const material=integer(primitive.material,0,materialInfo.length-1,"primitive material");
      const texcoords=primitive.texcoords===null?null:new Float32Array(vector(primitive.texcoords,count*2,"avatar texture coordinates"));
      if(materialInfo[material].base_color_texture!==null&&!texcoords)fail("Textured avatar needs UV0");
      if(!Array.isArray(primitive.indices)||primitive.indices.length%3)fail("Invalid avatar triangles");
      triangles+=primitive.indices.length/3;
      if(triangles>LIMITS.triangles)fail("Avatar triangle bound exceeded");
      const indices=new Uint32Array(primitive.indices.map(value=>integer(value,0,count-1,"triangle index")));
      let joints=null,weights=null;
      if(primitive.skin!==null){
        joints=new Uint16Array(vector(primitive.joints,count*4,"skin joint indices").map(value=>integer(value,0,skin.joints.length-1,"skin joint index")));
        weights=new Float32Array(vector(primitive.weights,count*4,"skin weights",1));
        for(let i=0;i<weights.length;i+=4)if(weights.slice(i,i+4).some(value=>value<0)||Math.abs(weights[i]+weights[i+1]+weights[i+2]+weights[i+3]-1)>1e-4)fail("Invalid normalized skin weights");
      }else if(primitive.joints!==null||primitive.weights!==null)fail("Unskinned attachment has skin attributes");
      return {node:primitive.node,skin:primitive.skin,positions,normals,texcoords,joints,weights,indices,material};
    });
    if(vertices>LIMITS.vertices||triangles>LIMITS.triangles)fail("Avatar geometry exceeds supported bounds");
    return {nodes:byId,order,root_node:rig.root_node,bindings,skin:{joints:[...skin.joints],inverseBind},primitives:meshes,framing:prepareFraming(meshes,skin.joints.length),materials:materialInfo,textures:textureInfo,stats:{vertices,triangles,pixels},basis,restRelative,rest,restInverse,roles};
  }
  function retarget(asset, body) {
    if(asset.restRelative)return retargetRest(asset,body);
    const nodes=new Map();
    for(const id of asset.order){
      const node=asset.nodes.get(id),binding=asset.bindings.get(id);
      if(binding){if(!body.frames[binding.body])fail("Missing simulated bone frame");nodes.set(id,multiply(body.frames[binding.body],binding.offset));}
      else nodes.set(id,node.parent===null?multiply(asset.basis,node.local):multiply(nodes.get(node.parent),node.local));
    }
    const palette=asset.skin.joints.map((id,i)=>multiply(nodes.get(id),asset.skin.inverseBind[i]));
    return {nodes,palette};
  }
  function retargetRest(asset,body){
    const nodes=new Map(),desired=new Map(),torso=body.frames.torso;
    if(!torso)fail("Missing simulated torso frame");
    const restTorso=asset.rest.get(asset.roles.get("torso"));
    const pivot=transform(restTorso.slice(12,15).map(v=>-v));
    desired.set(asset.root_node,multiply(multiply(torso,pivot),asset.rest.get(asset.root_node)));
    // Joint rotations come from physics; pivot offsets remain the avatar's rest geometry.
    for(const role of BODY_FRAMES.slice(1)){
      const id=asset.roles.get(role),rest=asset.rest.get(id),frame=body.frames[role];
      if(!frame)fail("Missing simulated bone frame");
      const rotation=frame.slice();rotation[12]=rotation[13]=rotation[14]=0;
      const result=multiply(rotation,rest);
      let position=torso.slice(12,15);
      if(role!=="torso"){
        const parent=role.includes("elbow")?role.replace("elbow","shoulder"):role.includes("knee")?role.replace("knee","hip"):"torso";
        const parentId=asset.roles.get(parent);
        position=point(multiply(desired.get(parentId),asset.restInverse.get(parentId)),rest.slice(12,15));
      }
      result.splice(12,3,...position);desired.set(id,result);
    }
    for(const id of asset.order){const node=asset.nodes.get(id);nodes.set(id,desired.get(id) || multiply(node.parent===null?asset.basis:nodes.get(node.parent),node.local));}
    return {nodes,palette:asset.skin.joints.map((id,i)=>multiply(nodes.get(id),asset.skin.inverseBind[i]))};
  }
  function posedGeometry(asset, pose) {
    return asset.primitives.map(primitive=>{
      const positions=new Float32Array(primitive.positions.length),normals=new Float32Array(primitive.normals.length);
      for(let vertex=0;vertex<positions.length/3;vertex++){
        let m=pose.nodes.get(primitive.node);
        if(primitive.skin!==null){m=new Array(16).fill(0);for(let j=0;j<4;j++){const weight=primitive.weights[vertex*4+j],bone=pose.palette[primitive.joints[vertex*4+j]];for(let k=0;k<16;k++)m[k]+=weight*bone[k];}}
        positions.set(point(m,primitive.positions.subarray(vertex*3,vertex*3+3)),vertex*3);
        normals.set(normalVector(m,primitive.normals.subarray(vertex*3,vertex*3+3)),vertex*3);
      }
      return {...primitive,positions,normals};
    });
  }
  const emptyBounds=()=>({min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]});
  function includePoint(bounds,xyz){for(let i=0;i<3;i++){const value=finite(xyz[i],"avatar framing coordinate",1e12);bounds.min[i]=Math.min(bounds.min[i],value);bounds.max[i]=Math.max(bounds.max[i],value);}}
  function boundsCorners(bounds){return Array.from({length:8},(_,mask)=>[0,1,2].map(i=>(mask&(1<<i))?bounds.max[i]:bounds.min[i]));}
  function prepareFraming(meshes,joints){
    const skin=Array.from({length:joints},emptyBounds),statics=new Map();
    for(const mesh of meshes){
      if(mesh.skin===null&&!statics.has(mesh.node))statics.set(mesh.node,emptyBounds());
      for(let vertex=0;vertex<mesh.positions.length/3;vertex++){
        const xyz=mesh.positions.subarray(vertex*3,vertex*3+3);
        if(mesh.skin===null)includePoint(statics.get(mesh.node),xyz);
        else for(let influence=0;influence<4;influence++)if(mesh.weights[vertex*4+influence]>0)includePoint(skin[mesh.joints[vertex*4+influence]],xyz);
      }
    }
    return {skin:skin.map(bounds=>Number.isFinite(bounds.min[0])?boundsCorners(bounds):[]),statics:[...statics].map(([node,bounds])=>({node,corners:boundsCorners(bounds)}))};
  }
  function posedBounds(asset,pose,world=null,matrices=null,snapshot=null){
    const bounds=emptyBounds();let corners=0;
    const add=(points,m)=>{
      for(const xyz of points){
        const w=m[3]*xyz[0]+m[7]*xyz[1]+m[11]*xyz[2]+m[15];
        if(!(w>1e-12))fail("Unsupported projective avatar framing");
        includePoint(bounds,point(m,xyz));corners++;
      }
    };
    // Positive skin weights place every vertex inside its transformed influence boxes.
    asset.framing.skin.forEach((points,index)=>add(points,pose.palette[index]));
    for(const row of asset.framing.statics)add(row.corners,pose.nodes.get(row.node));
    if(world!==null){
      if(!Array.isArray(matrices)||matrices.length!==world.meshes.length)fail("Incomplete framed world transforms");
      world.meshes.forEach((mesh,index)=>{if(mesh.id!==null)add(mesh.framing,matrices[index]);});
    }
    if(snapshot!==null){
      const state=snapshot?.profile==="shared-exercise-state-v1"?snapshot.engine_state:snapshot,segments=state?.human?.segments||[];
      if(!Array.isArray(segments)||segments.length>32)fail("Body framing exceeds its segment bound");
      for(const segment of segments){const radius=finite(segment.radius,"body framing radius",1);if(radius<=0)fail("Invalid body framing radius");for(const endpoint of [segment.a,segment.b]){const xyz=vector(endpoint,3,"body framing point");includePoint(bounds,xyz.map(v=>v-radius));includePoint(bounds,xyz.map(v=>v+radius));}}
    }
    if(!corners)fail("Avatar has no frameable geometry");
    for(let i=0;i<3;i++){const pad=Math.max(1e-5,Math.max(Math.abs(bounds.min[i]),Math.abs(bounds.max[i]))*2e-6);bounds.min[i]-=pad;bounds.max[i]+=pad;}
    return {min:bounds.min,max:bounds.max,corners};
  }
  function cameraViews(config, snapshot, camera={}, width=1080, height=420, framing=null) {
    integer(width,3,2048,"avatar viewport width");integer(height,1,2048,"avatar viewport height");
    const state=snapshot?.profile==="shared-exercise-state-v1"?snapshot.engine_state:snapshot;
    const yaw=finite(camera.yaw??.5,"camera yaw",100),zoom=finite(camera.zoom??1,"camera zoom",3);
    if(zoom<1)fail("Invalid camera zoom");
    const moving=(state?.objects||[]).filter(object=>object.role!=="static");let center=camera.follow&&moving.length
      ? [0,1,2].map(i=>moving.reduce((sum,object)=>sum+finite(object.position[i],"camera object position"),0)/moving.length)
      :config.bounds.map(value=>value/2);
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(.5),sp=Math.sin(.5),pane=width/3;
    let scale=zoom*Math.min(pane*.84/Math.max(config.bounds[0],config.bounds[1]),height*.76/Math.max(...config.bounds));
    const views=[{name:"Front: x / height",x:[1,0,0],y:[0,0,-1],depth:[0,1,0]},
      {name:"Top: x / depth",x:[1,0,0],y:[0,-1,0],depth:[0,0,1]},
      {name:"3D: coupled world",x:[cy,-sy,0],y:[sy*sp,cy*sp,-cp],depth:[sy*cp,cy*cp,sp]}];
    let half=null;
    if(framing!==null){
      const low=vector(framing.min,3,"camera lower bound",1e12),high=vector(framing.max,3,"camera upper bound",1e12);
      if(low.some((value,i)=>value>high[i]))fail("Camera framing bounds are reversed");
      if(!camera.follow)for(let i=0;i<3;i++){low[i]=Math.min(0,low[i]);high[i]=Math.max(config.bounds[i],high[i]);}
      center=low.map((value,i)=>(value+high[i])/2);half=low.map((value,i)=>(high[i]-value)/2);
      scale=zoom*Math.min(...views.flatMap(view=>[pane*.84/Math.max(1e-6,2*dot(view.x.map(Math.abs),half)),height*.76/Math.max(1e-6,2*dot(view.y.map(Math.abs),half))]));
    }
    return views.map((view,index)=>{
      const depth=half?Math.max(512,dot(view.depth.map(Math.abs),half)*1.1):512;
      const x=view.x.map(value=>value*2*scale/pane),y=view.y.map(value=>-value*2*scale/height),z=view.depth.map(value=>-value/depth);
      return {name:view.name,viewport:[Math.round(index*pane),0,Math.round((index+1)*pane)-Math.round(index*pane),height],
        matrix:[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,center),-.08-dot(y,center),-dot(z,center),1]};
    });
  }
  const srgbToLinear=value=>value<=.04045?value/12.92:((value+.055)/1.055)**2.4;
  function worldColor(value) {
    if(value===undefined)value="#C5A059";
    if(typeof value!=="string"||!/^#[0-9a-f]{6}$/i.test(value))fail("Unsupported world display color");
    return [1,3,5].map(index=>srgbToLinear(parseInt(value.slice(index,index+2),16)/255)).concat(1);
  }
  function primitiveGeometry(shape) {
    const positions=[],normals=[],indices=[];
    if(shape.shape==="box"){
      const size=vector(shape.size,3,"box size",100);if(size.some(value=>value<=0))fail("Invalid box size");
      const faces=[[[1,0,0],[[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]]],[[-1,0,0],[[-1,1,-1],[-1,-1,-1],[-1,-1,1],[-1,1,1]]],
        [[0,1,0],[[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]]],[[0,-1,0],[[-1,-1,1],[-1,-1,-1],[1,-1,-1],[1,-1,1]]],
        [[0,0,1],[[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]],[[0,0,-1],[[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]]];
      for(const [normal,points] of faces){const start=positions.length/3;for(const p of points){positions.push(...p.map((v,i)=>v*size[i]/2));normals.push(...normal);}indices.push(start,start+1,start+2,start,start+2,start+3);}
    }else if(["sphere","capsule"].includes(shape.shape)){
      const radius=finite(shape.radius,"rigid radius",100),half=shape.shape==="capsule"?finite(shape.length,"capsule length",100)/2:0;
      if(radius<=0||half<0)fail("Invalid rigid round shape");
      const rings=[];for(let i=0;i<=8;i++){rings.push([Math.PI*i/8,i<=4?half:-half]);if(i===4&&half>0)rings.push([Math.PI/2,-half]);}
      for(const [theta,offset] of rings)for(let j=0;j<=12;j++){const phi=j*Math.PI/6,n=[Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)];positions.push(n[0]*radius,n[1]*radius,n[2]*radius+offset);normals.push(...n);}
      for(let i=0;i<rings.length-1;i++)for(let j=0;j<12;j++){const a=i*13+j,b=(i+1)*13+j;indices.push(a,b,a+1,a+1,b,b+1);}
    }else fail("Unsupported rigid geometry in avatar view");
    return {positions,normals,indices};
  }
  function prepareWorld(config) {
    const meshes=[],ids=new Set();let vertices=0,triangles=0;
    for(const object of config.objects){
      if(typeof object.id!=="string"||ids.has(object.id))fail("Invalid world object identity");ids.add(object.id);
      const parts=object.shape==="compound"?object.parts:[object];
      if(!Array.isArray(parts)||!parts.length||parts.length>32)fail("Invalid compound geometry");
      const positions=[],normals=[],indices=[];
      for(const part of parts){
        const geometry=primitiveGeometry(part),offset=positions.length/3;
        let local=identity();
        if(object.shape==="compound"){
          const [w,x,y,z]=quaternion(part.quaternion,"compound quaternion");
          local=transform(vector(part.position,3,"compound offset",100),[x,y,z,w]);
        }
        for(let i=0;i<geometry.positions.length;i+=3){positions.push(...point(local,geometry.positions.slice(i,i+3)));normals.push(...[0,1,2].map(r=>local[r]*geometry.normals[i]+local[4+r]*geometry.normals[i+1]+local[8+r]*geometry.normals[i+2]));}
        indices.push(...geometry.indices.map(index=>index+offset));
      }
      vertices+=positions.length/3;triangles+=indices.length/3;
      if(vertices>200000||triangles>250000)fail("World render geometry exceeds supported bounds");
      const bounds=emptyBounds();for(let i=0;i<positions.length;i+=3)includePoint(bounds,positions.slice(i,i+3));
      meshes.push({id:object.id,positions:new Float32Array(positions),normals:new Float32Array(normals),indices:new Uint32Array(indices),color:worldColor(object.color),framing:boundsCorners(bounds)});
    }
    const span=Math.max(...config.bounds)*2,n=config.floor_normal,q=n[2]<-.999999?[1,0,0,0]:normalize([-n[1],n[0],0,1+n[2]]);
    meshes.push({id:null,positions:new Float32Array([-span,-span,0,span,-span,0,span,span,0,-span,span,0]),normals:new Float32Array([0,0,1,0,0,1,0,0,1,0,0,1]),indices:new Uint32Array([0,1,2,0,2,3]),color:worldColor("#E6D8C8")});
    return {meshes,floor_matrix:transform(config.origin,q),ids};
  }
  function worldMatrices(world, snapshot) {
    const state=snapshot?.profile==="shared-exercise-state-v1"?snapshot.engine_state:snapshot;
    if(!Array.isArray(state?.objects)||state.objects.length!==world.ids.size)fail("Missing rigid world state");
    const matrices=new Map();
    for(const object of state.objects){
      if(!world.ids.has(object.id)||matrices.has(object.id))fail("Mismatched rigid world state");
      const [w,x,y,z]=quaternion(object.quaternion,"rigid quaternion");
      matrices.set(object.id,transform(vector(object.position,3,"rigid position"),[x,y,z,w]));
    }
    return world.meshes.map(mesh=>mesh.id===null?world.floor_matrix:matrices.get(mesh.id));
  }
  const api=Object.freeze({LIMITS,BODY_FRAMES,identity,multiply,transform,axisRotation,point,inverse,normalVector,compileScene,bodyFrames,prepareAsset,retarget,posedGeometry,posedBounds,cameraViews,prepareWorld,worldMatrices,srgbToLinear});
  root.WeftAvatarRetarget=api;
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
})(globalThis);
