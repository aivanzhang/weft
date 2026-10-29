"""Development-only compact-object tracker; immutable appearance validates every flow proposal."""

from copy import deepcopy
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from weft.generic_vision import _overlap, _preview, validate_config
from weft.io import canonical_hash, digest, write_json
from weft.scene_vision import _camera_sample
from weft.video_io import webm_video_frame_count


ANGLES = tuple(range(-150, 181, 30))
SCALES = (.85, 1., 1.15)
BLURS = (0., 1.5, 3.)


class FlowValidatedAppearance:
    def __init__(self, gray, box, settings):
        self.box = tuple(box)
        self.settings = dict(settings)
        self.gap = 0
        self.lost = False
        self.velocity = np.zeros(2)
        x, y, w, h = box
        if min(w, h) < 6 or x < 0 or y < 0 or x+w > gray.shape[1] or y+h > gray.shape[0]:
            raise ValueError('Initialized object exceeds source bounds')
        template = gray[y:y+h,x:x+w]
        if template.std() < 8:
            raise ValueError('Object lacks contrast for immutable validation')
        self.side = math.ceil(math.hypot(w,h)*max(SCALES)) + 4
        # A central support mask is authored, not a claim of foreground segmentation.
        self.center = np.array([x+w/2,y+h/2])
        self.bank = []
        square = np.full((self.side,self.side), int(np.median(template)),np.uint8)
        mask = np.zeros_like(square)
        px,py = (self.side-w)//2,(self.side-h)//2
        square[py:py+h,px:px+w] = template
        cv2.ellipse(mask,(round(self.side/2),round(self.side/2)),(max(3,w//2-2),max(3,h//2-2)),0,0,360,255,-1)
        for angle in ANGLES:
            for scale in SCALES:
                transform = cv2.getRotationMatrix2D((self.side/2,self.side/2),angle,scale)
                rotated = cv2.warpAffine(square,transform,(self.side,self.side),borderMode=cv2.BORDER_REFLECT)
                support = cv2.warpAffine(mask,transform,(self.side,self.side),flags=cv2.INTER_NEAREST)
                for sigma in BLURS:
                    image = cv2.GaussianBlur(rotated,(0,0),sigma) if sigma else rotated
                    if image[support>0].std() >= 8:
                        self.bank.append((sigma,image,support,angle,scale))
        if not self.bank:
            raise ValueError('No supported immutable appearance templates')
        self._refresh_flow(gray)

    def _refresh_flow(self,gray):
        x,y,w,h = self.box
        mask=np.zeros_like(gray)
        cv2.ellipse(mask,(round(x+w/2),round(y+h/2)),(max(3,w//2-2),max(3,h//2-2)),0,0,360,255,-1)
        self.reference=gray.copy()
        self.points=cv2.goodFeaturesToTrack(gray,maxCorners=40,qualityLevel=.02,minDistance=3,mask=mask)

    def _flow_proposal(self,gray):
        if self.points is None or len(self.points)<4:
            return None,{'status':'insufficient_features'}
        forward,valid,_=cv2.calcOpticalFlowPyrLK(self.reference,gray,self.points,None,winSize=(21,21),maxLevel=3)
        if forward is None:
            return None,{'status':'flow_failed'}
        backward,back_valid,_=cv2.calcOpticalFlowPyrLK(gray,self.reference,forward,None,winSize=(21,21),maxLevel=3)
        if backward is None:
            return None,{'status':'backward_flow_failed'}
        errors=np.linalg.norm(backward-self.points,axis=2).ravel()
        keep=(valid.ravel()==1)&(back_valid.ravel()==1)&np.isfinite(errors)&(errors<=1.5)
        if keep.sum()<4:
            return None,{'status':'insufficient_consistent_features','consistent_features':int(keep.sum())}
        initial=self.points[keep].reshape(-1,2);target=forward[keep].reshape(-1,2)
        affine,inliers=cv2.estimateAffinePartial2D(initial,target,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=200)
        if affine is None or inliers is None or inliers.sum()<4:
            return None,{'status':'inconsistent_affine'}
        scale=math.hypot(affine[0,0],affine[1,0]);rotation=math.degrees(math.atan2(affine[1,0],affine[0,0]))
        proposed=affine[:,:2]@self.center+affine[:,2]
        coverage=float(np.prod(np.ptp(initial[inliers.ravel()==1],axis=0))/(self.box[2]*self.box[3]))
        plausible=(.7<=scale<=1.3 and abs(rotation)<=60 and coverage>=.1
                   and np.linalg.norm(proposed-self.center)<=self.settings['search_radius_px']*2*min(3,self.gap+1))
        return (proposed if plausible else None),{'status':'proposal_only' if plausible else 'implausible_affine',
                    'inliers':int(inliers.sum()),'coverage':coverage,'scale':scale,'rotation_degrees':rotation,
                    'median_forward_backward_error_px':float(np.median(errors[keep]))}

    def locate(self,gray):
        if self.lost:
            return {'status':'lost','score':None,'margin':None,'box':None,'flow':{'status':'lost'}}
        flow,diagnostics=self._flow_proposal(gray)
        predicted=self.center+self.velocity*min(3,self.gap+1)
        if flow is not None:
            predicted=flow
        radius=self.settings['search_radius_px'];half=self.side/2
        left,top=max(0,math.floor(predicted[0]-half-radius)),max(0,math.floor(predicted[1]-half-radius))
        right,bottom=min(gray.shape[1],math.ceil(predicted[0]+half+radius)),min(gray.shape[0],math.ceil(predicted[1]+half+radius))
        if right-left<self.side or bottom-top<self.side:
            return {'status':'out_of_frame','score':None,'margin':None,'box':None,'flow':diagnostics}
        patch=gray[top:bottom,left:right]
        blurred={sigma:cv2.GaussianBlur(patch,(0,0),sigma) if sigma else patch for sigma in BLURS}
        aggregate=np.full((patch.shape[0]-self.side+1,patch.shape[1]-self.side+1),-1.,np.float32)
        best_model=None;best_score=-1
        for sigma,template,mask,angle,scale in self.bank:
            scores=cv2.matchTemplate(blurred[sigma],template,cv2.TM_CCOEFF_NORMED,mask=mask)
            scores=np.nan_to_num(scores,nan=-1.,posinf=-1.,neginf=-1.)
            # OpenCV's masked normalization is undefined for constant comparison patches.
            scores=np.where((scores>=-1.)&(scores<=1.00001),np.minimum(scores,1.),-1.)
            value=float(scores.max())
            if value>best_score:
                best_score=value;best_model={'rotation_degrees':angle,'scale':scale,'blur_sigma':sigma}
            np.maximum(aggregate,scores,out=aggregate)
        _,best,_,location=cv2.minMaxLoc(aggregate)
        secondary=aggregate.copy();sx,sy=location;w,h=self.box[2:]
        dx,dy=max(3,w//3),max(3,h//3)
        secondary[max(0,sy-dy):sy+dy+1,max(0,sx-dx):sx+dx+1]=-1
        margin=float(best-secondary.max())
        status='tracked' if best>=self.settings['minimum_score'] and margin>=self.settings['minimum_margin'] else ('ambiguous' if best>=self.settings['minimum_score'] else 'not_detected')
        center=np.array([left+sx+half,top+sy+half])
        box=[float(center[0]-w/2),float(center[1]-h/2),w,h] if status=='tracked' else None
        return {'status':status,'score':float(best),'margin':margin,'box':box,'flow':diagnostics,'bank_match':best_model}

    def commit(self,candidate,gray):
        if candidate['status']=='tracked':
            new=candidate['box'];center=np.array([new[0]+new[2]/2,new[1]+new[3]/2])
            self.velocity=.5*self.velocity+.5*(center-self.center)/(self.gap+1)
            self.center=center;self.box=tuple(new);self.gap=0
            self._refresh_flow(gray)
        else:
            self.gap+=1
            if self.gap>self.settings['max_gap_frames']:
                self.lost=True


def track_prototype(video,out,config):
    """Produce motion evidence only; camera and reconstruction eligibility are never relaxed."""
    video,out=Path(video),Path(out);config=validate_config(config)
    if (config.get('decode_interval') or config['initialization_frame'] != config['start_frame']
            or config['association_mode'] != 'independent'):
        raise ValueError('Flow prototype does not support intervals, delayed initialization, or joint association')
    if not video.is_file() or video.stat().st_size>100_000_000 or digest(video)!=config['source_sha256']:
        raise ValueError('Source is missing, oversized, or differs from pinned initialization')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'tracking-config.json',config)
    cap=cv2.VideoCapture(str(video));started=time.monotonic();rows=[];previews=[];camera=[];trackers=None
    try:
        fps,count,sw,sh=[cap.get(k) for k in (5,7,3,4)]
        if not all(math.isfinite(v) for v in (fps,count,sw,sh)) or not 1<=fps<=240 or not 1<=count<=1800 or count/fps>60 or not 64<=sw<=4096 or not 64<=sh<=2160:
            raise ValueError('Source exceeds bounded dimensions, frames or duration')
        sw,sh,count=int(sw),int(sh),int(count);scale=min(1,960/max(sw,sh));width,height=round(sw*scale),round(sh*scale)
        settings=dict(config,search_radius_px=max(8,round(config['search_radius_px']*scale)))
        boxes=[[round(v*scale) for v in obj['box_xywh']] for obj in config['objects']]
        if any(_overlap(a,b)>.5 for i,a in enumerate(boxes) for b in boxes[i+1:]):raise ValueError('Ambiguous initialized overlap')
        interval=min(config['end_frame'],count-1)-config['start_frame']+1
        if interval<1:raise ValueError('Empty authored interval')
        stride=max(1,math.ceil(interval/40));decoded=0
        while True:
            ok,source=cap.read()
            if not ok:break
            index=decoded;decoded+=1
            if decoded>1800 or decoded/fps>60 or time.monotonic()-started>120:raise ValueError('Prototype exceeded bounded envelope')
            if source.shape[:2]!=(sh,sw):raise ValueError('Source dimensions changed')
            if not config['start_frame']<=index<=config['end_frame']:continue
            frame=cv2.resize(source,(width,height),interpolation=cv2.INTER_AREA) if scale!=1 else source
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if trackers is None:
                trackers=[FlowValidatedAppearance(gray,box,settings) for box in boxes]
                reference=gray;mask=np.full(gray.shape,255,np.uint8)
                for x,y,w,h in boxes:mask[max(0,y-15):y+h+15,max(0,x-15):x+w+15]=0
                points=cv2.goodFeaturesToTrack(gray,maxCorners=120,qualityLevel=.01,minDistance=12,mask=mask)
                candidates=[{'status':'initialized','score':None,'margin':None,'box':box,'flow':{'status':'authored'}} for box in boxes]
            else:
                candidates=[tracker.locate(gray) for tracker in trackers]
                conflicts={i for i,a in enumerate(candidates) if a['box'] is not None and any(j!=i and b['box'] is not None and _overlap(a['box'],b['box'])>.5 for j,b in enumerate(candidates))}
                for i in conflicts:candidates[i].update(status='ambiguous_overlap',box=None)
                for tracker,candidate in zip(trackers,candidates):tracker.commit(candidate,gray)
            if (index-config['start_frame'])%max(1,round(fps/2))==0:camera.append(_camera_sample(reference,frame,points,index))
            overlay=frame.copy() if (index-config['start_frame'])%stride==0 else None
            for obj,candidate in zip(config['objects'],candidates):
                row={'id':f"{config['source_sha256']}:frame:{index}:object:{obj['id']}",'frame':index,'time_s':index/fps,'object_id':obj['id'],
                     'status':candidate['status'],'match_score':candidate['score'],'peak_margin':candidate['margin'],'box_xywh':candidate['box'],
                     'flow':candidate['flow'],'bank_match':candidate.get('bank_match'),'coordinate_system':'analysis_image_pixels',
                     'geometry_status':'authored compact ROI support; translated center, not recovered geometry'}
                if candidate['box'] is not None:
                    x,y,w,h=candidate['box'];row.update(x_px=x+w/2,y_px=y+h/2,x=(x+w/2)/width,y=(y+h/2)/width)
                    if overlay is not None:cv2.rectangle(overlay,(round(x),round(y)),(round(x+w),round(y+h)),(0,180,255),2)
                rows.append(row)
            if overlay is not None:
                cv2.putText(overlay,f'frame {index} | prototype motion evidence, not physics',(10,height-12),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,255),1)
                preview=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB));preview.thumbnail((640,640));previews.append(preview)
        if trackers is None:raise ValueError('No initialization frame decoded')
        audit={'metadata_frames':count,'decoded_frames':decoded}
        if count!=decoded:
            if video.suffix.lower() not in {'.webm','.mkv'}:raise ValueError('Incomplete decode')
            audit['encoded_video_frames']=webm_video_frame_count(video)
            if audit['encoded_video_frames']!=decoded:raise ValueError('Incomplete encoded frame decode')
        moving=any(row['status']=='moving_background' for row in camera)
        result={'schema_version':1,'profile':'appearance-flow-prototype-v1','source_sha256':config['source_sha256'],'width':width,'height':height,
                'source_width':sw,'source_height':sh,'fps':fps,'frame_count':count,'decoded_frames':decoded,'decode_audit':audit,
                'selected_interval':[config['start_frame'],min(config['end_frame'],decoded-1)],'observations':rows,
                'initialization':{'status':'authored','config_sha256':canonical_hash(config),'objects':config['objects']},
                'camera_check':{'result':'moving_background' if moving else 'not_verified_fixed','samples':camera},
                'physics_ready':False,'seconds':time.monotonic()-started,
                'assumptions':['Immutable rotated/scaled/blurred grayscale templates; ellipse support is authored, not segmentation',
                               'LK proposes a search center only; every published position requires immutable appearance and ambiguity validation',
                               'Appearance scores are not calibrated probabilities; template position need not equal physical center of mass',
                               'No physical force, geometry, depth, or camera recovery; camera motion still prevents fixed-camera physics']}
        write_json(out/'appearance-tracks.json',result);_preview(previews,out);return result
    except Exception as error:
        write_json(out/'failure.json',{'type':type(error).__name__,'message':str(error),'seconds':time.monotonic()-started})
        write_json(out/'partial-observations.json',rows);raise
    finally:cap.release()
