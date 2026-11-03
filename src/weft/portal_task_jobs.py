"""Existing portal worker jobs for authored task children and uninstalled policy reports."""

import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from uuid import uuid4

from weft.io import canonical_hash, local_time
from weft.lab_worker import atomic_json
from weft.shared_world_packaging import read_package_file, read_package_manifest, validate_package_layout, write_package_files

PROFILE = "portal-task-job-v1"
DIRECTORY = "task-job"
ACTIVE = {"queued", "running", "cancelling"}
TERMINAL = {"ready", "complete", "failed", "cancelled", "interrupted"}
MAX_JOB_BYTES = 650_000_000
MAX_FILES = 4096
IMPLEMENTATIONS = ("portal_task_jobs.py", "portal_task_contract.py")
ARTIFACTS = {"request.json", "start.json", "receipt.json", "report.json", "failure.json", "evaluation-protocol.json",
    "training/result.json", "training/evaluation.json", "training/policy.json", "training/failure.json", "training/manifest.json",
    "evaluation/result.json", "evaluation/failure.json", "evaluation/protocol.json", "evaluation/manifest.json"}
ARTIFACTS |= {"training/settings.json", "evaluation/nominal-start.json", "evaluation/plan.json"}
ARTIFACTS |= {f"training/generation-{index}.json" for index in range(1, 5)}
ARTIFACTS |= {f"evaluation/start-{index}.json" for index in range(1, 17)}
SCOPE = "Authored simulated task and bounded development comparisons. Original world/scorer unchanged; no policy installation or physical validation."


def _require(condition, message):
    if not condition:
        raise ValueError("Portal task job: " + message)


def _check(deadline, child=None):
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "finite deadline required")
    if time.monotonic() >= deadline:
        raise TimeoutError("Task job exceeded its aggregate deadline")
    if child is not None and any((Path(child)/n).exists() or (Path(child)/n).is_symlink()
                                for n in ("cancel-request.json", DIRECTORY+"/cancel-request.json")):
        raise InterruptedError("Cancelled; original source and partial results are preserved")


def _json(raw, maximum=128 << 10):
    from weft.spatial_scene_inputs import _json as strict_json
    return strict_json(raw, maximum)


def _bytes(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    _require(len(raw) <= 128 << 10, "small job metadata exceeds 128 KiB")
    return raw


def _read(root, name, maximum=128 << 10, pin=None):
    return read_package_file(root, name, max_bytes=maximum, expected_sha256=pin)


def _directory(path):
    path = Path(path).absolute()
    _require(".." not in path.parts and path.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)), "safe existing directory required")
    return path


def _parent(parent, operation):
    parent = _directory(parent)
    _require(re.fullmatch("[0-9a-f]{32}", parent.name), "parent project ID differs")
    files = {n: _read(parent,n) for n in ("job.json","status.json")}
    job, state = (_json(files[n]) for n in ("job.json","status.json"))
    _require(job.get("id") == parent.name and state.get("status") == "ready", "ready parent project required")
    for name in ("perception/status.json","equipment/status.json","moving-equipment/status.json","muse/status.json"):
        path = parent/name
        if path.exists() or path.is_symlink():
            files[name] = _read(parent,name)
            _require(_json(files[name]).get("status") not in ACTIVE | {"preflighting","analyzing","preparing_tracking"}, "parent work is active")
    _require(not (parent/"cancel-request.json").exists(), "parent cancellation is preserved")
    relative = state.get("lab","lab")
    _require(type(relative) is str and len(relative) <= 256 and len(Path(relative).parts) <= 8, "bounded selected lab path required")
    validate_package_layout([relative+"/lab-config.json", relative+"/run/manifest.json"])
    lab = _directory(parent/relative)
    config_name, manifest_name = relative+"/lab-config.json",relative+"/run/manifest.json"
    files[config_name] = _read(parent,config_name,64 << 10)
    files[manifest_name] = _read(parent,manifest_name,1 << 20)
    config, manifest = _json(files[config_name]),_json(files[manifest_name],1 << 20)
    expected = "shared_world_exploration" if operation == "create" else "moving_equipment_task"
    _require(config.get("control_profile") == expected and config.get("manifest_sha256") == sha256(files[manifest_name]).hexdigest(), "selected parent lab profile or seal differs")
    members = manifest.get("files")
    _require(type(members) is dict and len(members) <= 2080, "bounded parent manifest required")
    marker_name = "exploration-lab.json" if operation == "create" else "task-lab.json"
    marker_path = relative+"/run/"+marker_name
    _require(marker_name in members, "parent contract is not sealed")
    files[marker_path] = _read(parent,marker_path,128 << 10,members[marker_name])
    marker = _json(files[marker_path])
    _require(marker.get("scene_sha256") == config.get("scene_sha256"), "parent scene identity differs")
    if operation == "train_evaluate":
        name = relative+"/run/placement-task.json"
        _require("placement-task.json" in members, "task is not sealed")
        files[name] = _read(parent,name,64 << 10,members["placement-task.json"])
        task = _json(files[name])
        _require(canonical_hash(task) == marker.get("task_sha256") == config.get("task_sha256"), "parent task identity differs")
    else:
        task = None
    return job, state, relative, config, marker, task, files


def _identity(operation, payload, parent_id, pins):
    return canonical_hash({"operation":operation,"request":{k:v for k,v in payload.items() if k != "request_id"},
                           "parent_id":parent_id,"parent_files":pins})


def _queue(root, parent_id, payload, operation, *, lease_fd, existing_jobs, stored_bytes, reserved_bytes, pending_jobs, deadline):
    from weft import portal_task_contract as contract
    from weft.portal import MAX_JOBS, MAX_STORAGE
    _check(deadline)
    root = _directory(root)
    _require(type(parent_id) is str and re.fullmatch("[0-9a-f]{32}",parent_id), "exact parent ID required")
    _require(type(lease_fd) is int and not (root/"portal.lock").is_symlink(), "existing portal lease required")
    descriptor, lock = os.fstat(lease_fd),(root/"portal.lock").stat()
    _require((descriptor.st_dev,descriptor.st_ino)==(lock.st_dev,lock.st_ino), "lease belongs to another portal")
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent = root/parent_id
    job,state,relative,config,marker,task,snapshots = _parent(parent,operation)
    _require(type(existing_jobs) is dict and len(existing_jobs) <= MAX_JOBS and existing_jobs.get(parent_id)==job, "parent differs from bounded inventory")
    request = (contract.validate_task_create_request(payload) if operation == "create" else
               contract.validate_task_learning_request(payload,horizon_steps=task["interval"]["horizon_steps"]))
    if operation == "create":
        _require(request["parent_manifest_sha256"] == config["manifest_sha256"], "requested parent seal differs")
    else:
        _require(request["task_sha256"] == config["task_sha256"], "requested task differs")
    pins = {n:sha256(v).hexdigest() for n,v in snapshots.items()}
    identity = _identity(operation,request,parent_id,pins)
    for key, other in existing_jobs.items():
        _check(deadline)
        if other.get("request_id") == request["request_id"] or other.get("task_action_identity") == identity:
            _require(other.get("task_action_identity") == identity, "request ID already binds a different action")
            saved = _json(_read(root/key,DIRECTORY+"/request.json"))
            _require(saved.get("action_identity") == identity and saved.get("parent_files") == pins, "existing action changed; no retry")
            return other
    _require(all(type(x) is int and x >= 0 for x in (stored_bytes,reserved_bytes,pending_jobs))
             and len(existing_jobs)<MAX_JOBS and pending_jobs<4
             and stored_bytes+reserved_bytes+MAX_JOB_BYTES<=MAX_STORAGE,"portal cannot reserve this complete bounded child")
    child_id = uuid4().hex; stage=root/(".task-job-"+child_id)
    stage.mkdir(mode=0o700)
    request_doc={"schema_version":1,"profile":PROFILE,"operation":operation,"child_id":child_id,"parent_id":parent_id,
        "parent_lab":relative,"parent_files":pins,"payload":request,"action_identity":identity,"api_calls_authorized":0,
        "automatic_policy_installation":False}
    request_raw=_bytes(request_doc)
    child={"id":child_id,"kind":"task_job","task_operation":operation,"name":("Authored task" if operation=="create" else "Training and evaluation"),
        "parent_id":parent_id,"source_sha256":marker["source_sha256"],"request_id":request["request_id"],"task_action_identity":identity,
        "task_request_sha256":sha256(request_raw).hexdigest(),"created_at":local_time(),"mode":"exercise","ai":False}
    if operation=="create": child["control_profile"]="moving_equipment_task"
    try:
        write_package_files(stage,{"job.json":_bytes(child),"status.json":_bytes({"status":"queued","stage":operation,"events":[],"scope":SCOPE}),
            DIRECTORY+"/request.json":request_raw,**{DIRECTORY+"/parent/"+n:v for n,v in snapshots.items()}})
        current=_parent(parent,operation)[-1]
        _require(current==snapshots,"parent changed before queue publication")
        _check(deadline)
        os.rename(stage,root/child_id)
        return child
    except Exception as error:
        write_package_files(stage,{DIRECTORY+"/publication-failure.json":_bytes({"status":"failed","error":str(error)[:1000],"preserved":True})})
        raise


def create_task_child(root,parent_id,payload,**options):
    return _queue(root,parent_id,payload,"create",**options)


def create_learning_child(root,parent_id,payload,**options):
    return _queue(root,parent_id,payload,"train_evaluate",**options)


def read_status(child):
    return _json(_read(child,"status.json",32 << 10))


def summary(child):
    """Only published small receipts; no model, source scan or admission on the request thread."""
    job=_json(_read(child,"job.json"));state=read_status(child)
    result={**state,"operation":job.get("task_operation"),"parent_id":job.get("parent_id"),"scope":SCOPE,"api_calls":0,"automatic_policy_installation":False}
    if state.get("report_sha256"):
        result["report"]=_json(_read(child,DIRECTORY+"/report.json",128 << 10,state["report_sha256"]))
    result["artifacts"]=list(state.get("artifacts",[]))
    return result


def reserved_growth(child):
    return MAX_JOB_BYTES if read_status(child).get("status") in ACTIVE else 0


def _tree(root, deadline):
    result={};total=0
    entries=0
    for path in root.rglob("*"):
        _check(deadline);entries+=1
        _require(entries<=MAX_FILES*2 and not path.is_symlink(),"linked or excessive worker artifact inventory")
        info=path.stat()
        if not stat.S_ISDIR(info.st_mode):
            _require(stat.S_ISREG(info.st_mode),"regular artifact required")
            total+=info.st_size;result[path.relative_to(root).as_posix()]=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
            _require(len(result)<=MAX_FILES and total<=MAX_JOB_BYTES,"complete task job exceeds its file/byte allowance")
    return result


def _bound_inputs(child, deadline):
    job=_json(_read(child,"job.json"));raw=_read(child,DIRECTORY+"/request.json",pin=job.get("task_request_sha256"));request=_json(raw)
    _require(set(request)=={"schema_version","profile","operation","child_id","parent_id","parent_lab","parent_files","payload","action_identity","api_calls_authorized","automatic_policy_installation"}
             and request["profile"]==PROFILE and type(request["schema_version"]) is int and request["schema_version"]==1
             and request["child_id"]==child.name==job.get("id") and request["operation"]==job.get("task_operation") in {"create","train_evaluate"}
             and request["parent_id"]==job.get("parent_id") and request["action_identity"]==job.get("task_action_identity")
             and request["action_identity"]==_identity(request["operation"],request["payload"],request["parent_id"],request["parent_files"])
             and type(request["api_calls_authorized"]) is int and request["api_calls_authorized"]==0
             and request["automatic_policy_installation"] is False and job.get("ai") is False
             and job.get("kind")=="task_job" and job.get("request_id")==request["payload"].get("request_id"),"closed task action binding differs")
    _require(type(request["parent_id"]) is str and re.fullmatch("[0-9a-f]{32}",request["parent_id"]),"parent must be one exact portal project ID")
    parent=_directory(child.parent/request["parent_id"])
    _require(parent.parent==child.parent,"parent must remain inside this portal")
    info=_parent(parent,request["operation"])
    _require(info[2]==request["parent_lab"] and job.get("source_sha256")==info[4]["source_sha256"]
             and {n:sha256(v).hexdigest() for n,v in info[-1].items()}==request["parent_files"],"parent changed after queue")
    for name,pin in request["parent_files"].items():
        _check(deadline,child);_read(child,DIRECTORY+"/parent/"+name,1 << 20,pin)
    return job,request,raw,parent,info


def _receipt_bindings(parent,info,task):
    _,_,relative,config,marker,_,snapshots=info
    return {"parent_project_id":parent.name,"parent_job_sha256":sha256(snapshots["job.json"]).hexdigest(),
        "parent_status_sha256":sha256(snapshots["status.json"]).hexdigest(),"parent_lab_config_sha256":sha256(snapshots[relative+"/lab-config.json"]).hexdigest(),
        "parent_manifest_sha256":config["manifest_sha256"],"source_sha256":marker["source_sha256"],
        "scene_sha256":task["binding"]["scene_sha256"],"family_sha256":task["binding"]["family_sha256"],
        "selection_sha256":task["binding"]["selection_sha256"],"action_contract_sha256":canonical_hash(task["action_contract"]),
        "task_sha256":canonical_hash(task),"scorer_sha256":task["scorer_sha256"]}


def execute_job(child, *, project=None):
    """One existing supervised, keyless worker; source admission and heavy copies stay here."""
    from weft import portal_task_contract as contract
    child=_directory(child);work=child/DIRECTORY;state=read_status(child)
    if state.get("status") in TERMINAL: return 0 if state["status"] in {"ready","complete"} else 1
    started=time.monotonic();deadline=started+180;stage="admission";artifacts=[]
    preexisting_start=(work/"start.json").exists() or (work/"start.json").is_symlink()
    try:
        _check(deadline,child)
        _require(state.get("status")=="queued" and not (work/"start.json").exists(),"started or partial task action cannot retry")
        write_package_files(work,{"start.json":_bytes({"profile":PROFILE,"started_at":local_time(),"api_calls":0,"retry":False})})
        atomic_json(child/"status.json",{"status":"running","stage":stage,"scope":SCOPE})
        job,request,request_raw,parent,info=_bound_inputs(child,deadline)
        job_raw=_read(child,"job.json")
        source=parent/info[2];source_signatures=_tree(source/"run",deadline)
        code={name:Path(__file__).with_name(name).read_bytes() for name in IMPLEMENTATIONS}
        write_package_files(work,{"implementation/"+n:v for n,v in code.items()})
        operation=request["operation"]
        if operation=="create":
            from weft.moving_equipment_task_lab import prepare_moving_equipment_task_lab
            payload=contract.validate_task_create_request(request["payload"])
            _require(payload["parent_manifest_sha256"]==info[3]["manifest_sha256"],"parent request pin differs")
            stage="create"
            built=prepare_moving_equipment_task_lab(source,payload["task"],child/"lab",parent_manifest_sha256=payload["parent_manifest_sha256"],max_seconds=90,deadline=deadline)
            config=_json(_read(child/"lab","lab-config.json",64 << 10))
            manifest_raw,members=read_package_manifest(child/"lab/run")
            _require(sha256(manifest_raw).hexdigest()==config["manifest_sha256"]==built["manifest_sha256"],"constructed task seal differs")
            task=_json(_read(child/"lab/run","placement-task.json",64 << 10,members["placement-task.json"]))
            report={"profile":PROFILE,"status":"ready","operation":"create","task_sha256":canonical_hash(task),
                "manifest_sha256":config["manifest_sha256"],"scene_sha256":config["scene_sha256"],"learned_policy_available":False,
                "task_success":False,"scope":SCOPE,"source_parent_id":parent.name}
            terminal={"status":"ready","lab":"lab","control_profile":"moving_equipment_task"}
        else:
            from weft.moving_equipment_task_lab import load_moving_equipment_task_lab
            from weft.moving_equipment_learning import train_moving_equipment_policy
            from weft.moving_equipment_evaluation import build_changed_start_protocol,evaluate_changed_starts
            admission=load_moving_equipment_task_lab(source/"run",info[3],max_seconds=30,deadline=deadline)
            task=admission.task
            payload=contract.validate_task_learning_request(request["payload"],horizon_steps=task["interval"]["horizon_steps"])
            _require(payload["task_sha256"]==canonical_hash(task),"admitted task differs from training request")
            protocol=None
            if payload["evaluation"] is not None:
                settings=payload["evaluation"]
                protocol=build_changed_start_protocol(admission,**{k:settings[k] for k in ("evaluation_seeds","position_jitter","velocity_jitter")})
                write_package_files(work,{"evaluation-protocol.json":_bytes(protocol)})
            stage="training";_check(deadline,child)
            atomic_json(child/"status.json",{"status":"running","stage":stage,"scope":SCOPE})
            trained=train_moving_equipment_policy(admission,work/"training",**payload["training"])
            _check(deadline,child)
            train_manifest,train_members=read_package_manifest(work/"training")
            _require(sha256(train_manifest).hexdigest()==trained["manifest_sha256"],"completed training manifest changed")
            training=_read(work/"training","result.json",128 << 10,train_members["result.json"])
            evaluation=_read(work/"training","evaluation.json",4 << 20,train_members["evaluation.json"])
            changed=None
            if protocol is not None:
                stage="evaluation";atomic_json(child/"status.json",{"status":"running","stage":stage,"scope":SCOPE})
                policy=_read(work/"training","policy.json",64 << 10,train_members["policy.json"])
                evaluated=evaluate_changed_starts(admission,protocol,policy,work/"evaluation",max_seconds=payload["evaluation"]["max_seconds"])
                _,eval_members=read_package_manifest(work/"evaluation")
                changed=_read(work/"evaluation","result.json",128 << 10,eval_members["result.json"])
                _require(canonical_hash(_json(changed))==canonical_hash(evaluated),"completed evaluation result changed")
            _check(deadline,child)
            report=contract.summarize_task_learning(training,evaluation,task_sha256=canonical_hash(task),
                scene_sha256=task["binding"]["scene_sha256"],evaluation_result_raw=changed)
            terminal={"status":"complete"}
        _check(deadline,child)
        stage="publication"
        receipt=contract.build_task_action_receipt(operation,payload,_receipt_bindings(parent,info,task))
        receipt_raw,report_raw=_bytes(receipt),_bytes(report)
        _require(_tree(source/"run",deadline)==source_signatures and _parent(parent,operation)[-1]==info[-1],"original source changed during task job")
        _require(_read(child,"job.json")==job_raw and _read(child,DIRECTORY+"/request.json")==request_raw and all(Path(__file__).with_name(n).read_bytes()==v for n,v in code.items()),"task request/runtime changed")
        inventory=_tree(child,deadline)
        _require(sum(v[2] for v in inventory.values())+(512 << 10)<=MAX_JOB_BYTES,"final receipts exceed child capacity")
        _check(deadline,child)
        write_package_files(work,{"receipt.json":receipt_raw,"report.json":report_raw})
        artifacts=[DIRECTORY+"/"+n for n in sorted(ARTIFACTS) if (work/n).is_file()]
        _check(deadline)
        atomic_json(child/"status.json",{**terminal,"stage":"complete","report_sha256":sha256(report_raw).hexdigest(),
            "receipt_sha256":sha256(receipt_raw).hexdigest(),"artifacts":artifacts,"seconds":time.monotonic()-started,
            "api_calls":0,"inference_calls":0,"automatic_policy_installation":False,"scope":SCOPE})
        return 0
    except Exception as error:
        failure={"status":"failed","stage":stage,"reason":f"{type(error).__name__}: {error}"[:1000],"scope":SCOPE,
                 "api_calls":0,"inference_calls":0,"retry":False,"partial_results_preserved":True}
        status="cancelled" if isinstance(error,InterruptedError) else "interrupted" if preexisting_start or state.get("status")=="running" else "failed"
        if not (work/"failure.json").exists():
            write_package_files(work,{"failure.json":_bytes(failure)})
        failure_raw=_read(work,"failure.json",2 << 20)
        prior=work/(stage+"/failure.json")
        if stage in {"training","evaluation"} and prior.is_file():
            failure_raw=_read(work,stage+"/failure.json",2 << 20)
        report=contract.summarize_task_failure(failure_raw,stage=stage)
        report_raw=_bytes(report)
        if not (work/"report.json").exists():
            write_package_files(work,{"report.json":report_raw})
        else:
            report_raw=_read(work,"report.json",128 << 10)
        artifacts=[DIRECTORY+"/"+n for n in sorted(ARTIFACTS) if (work/n).is_file()]
        atomic_json(child/"status.json",{"status":status,"stage":stage,"error":failure["reason"],"artifacts":artifacts,
            "report_sha256":sha256(report_raw).hexdigest(),"scope":SCOPE,"retry":False})
        return 1
