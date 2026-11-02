"""Read-only, bounded custody proof for failed source-reuse Studio jobs."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import stat
import subprocess
import time

from avatar_assets.studio_jobs import MAX_JSON, _local_time, _require


MAX_FILES = 20000
MAX_RECEIPTS = 512
MAX_RECEIPT_BYTES = 4 * 1024**2


def _receipt(path):
    _require(path.resolve()==path and path.is_file(),'Custody receipt path is unsafe or missing')
    fd=os.open(path,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size<=MAX_JSON,'Custody receipt is unsafe or oversized')
        raw=stream.read(MAX_JSON+1)
    _require(len(raw)<=MAX_JSON,'Custody receipt grew beyond its bound')
    value=json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite custody JSON')))
    _require(isinstance(value,dict),'Invalid custody receipt')
    return value,{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


@contextmanager
def custody_locks(paths):
    """Existing locks fence cooperating supervisors and direct pipeline executions without new files."""
    streams=[]
    try:
        for path in paths:
            _require(path.resolve()==path and path.is_file(),'Existing custody lock is missing or unsafe')
            fd=os.open(path,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            stream=os.fdopen(fd,'rb');streams.append(stream)
            _require(stat.S_ISREG(os.fstat(fd).st_mode),'Custody lock is not a regular file')
            try:fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('A supervisor or pipeline still owns its lock; no release') from None
        yield
    finally:
        for stream in reversed(streams):stream.close()


def _process_snapshot():
    """Inspect PID/session groups without command arguments, environment or signals to workers."""
    command=['/bin/ps','-axo','pid=,ppid=,pgid=']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
    data=bytearray();deadline=time.monotonic()+2
    try:
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdout.fileno(),False)
            selector.register(process.stdout,selectors.EVENT_READ)
            while selector.get_map():
                remaining=deadline-time.monotonic()
                _require(remaining>0,'OS custody inspection exceeded two seconds')
                for key,_ in selector.select(remaining):
                    chunk=os.read(key.fd,65536)
                    if not chunk:selector.unregister(key.fileobj)
                    else:
                        data.extend(chunk)
                        _require(len(data)<=512*1024,'OS custody snapshot exceeds512KiB')
        _require(process.wait(timeout=max(.01,deadline-time.monotonic()))==0,'OS custody inspection failed')
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=1)
        process.stdout.close()
    rows=[]
    for line in data.decode('ascii').splitlines():
        fields=line.split()
        _require(len(fields)==3 and all(field.isdigit() for field in fields),'Invalid OS custody snapshot')
        rows.append(tuple(int(field) for field in fields))
    _require(rows and len(rows)<=MAX_FILES,'OS process inventory is empty or oversized')
    return rows,hashlib.sha256(data).hexdigest()


def assert_absent(pids):
    rows,checksum=_process_snapshot()
    matched=[{'pid':pid,'parent_pid':parent,'process_group':group} for pid,parent,group in rows
             if pid in pids or parent in pids or group in pids]
    _require(not matched,'A recorded owner, child or process group is still present; no release')
    return {'recorded_local':_local_time(time.time()),'known_pid_count':len(pids),
            'os_snapshot_sha256':checksum,'matched_processes':[],
            'scope':'Recorded PIDs, direct children and process groups are absent; no worker was signalled.'}


def collect(root,output,invocation,owner_pid,owner_identity):
    root,output,invocation=map(Path,(root,output,invocation))
    _require(output.resolve()==output and output.is_dir() and output.is_relative_to(root/'outputs'),'Unsafe inspected job output')
    _require(invocation.resolve()==invocation and invocation.is_dir() and invocation.is_relative_to(root/'.local'),'Unsafe inspected invocation')
    owner,_=_receipt(invocation/'owner.json')
    _require(owner.get('pid')==owner_pid and owner.get('start_identity')==owner_identity,
             'Sealed supervisor owner differs from the retained lease')
    child,_=_receipt(invocation/'worker-job.json')
    _require(child.get('process_exit_verified') is True
             and type(child.get('exit_code')) is int and child['exit_code'] in (0,1) and child.get('status')=='finished'
             and child.get('manual_stop_required') is False,'Controller child has no sealed safe exit receipt')
    terminal_paths={invocation/'result.json',invocation/'supervisor-result.json'}
    paths={invocation/'owner.json',invocation/'worker-job.json',invocation/'invocation.json',*terminal_paths}
    deadline=time.monotonic()+8;files=0
    for folder in (output,invocation):
        for directory,dirs,names in os.walk(folder,followlinks=False):
            _require(time.monotonic()<deadline,'Custody receipt inventory exceeded eight seconds')
            files+=len(dirs)+len(names);_require(files<=MAX_FILES,'Custody file inventory exceeds its bound')
            for name in dirs+names:
                _require(not (Path(directory)/name).is_symlink(),'Custody inventory contains a symlink')
            for name in names:
                if name=='job.json' or name.endswith(('-handle.json','-monitor.json','-process.json','-job.json')):
                    paths.add(Path(directory)/name)
                    _require(len(paths)<=MAX_RECEIPTS,'Too many custody receipts')
    records=[];pids={owner_pid};closed=set();exits={};handles=[];total=0
    for path in sorted(paths):
        value,record=_receipt(path)
        total+=record['bytes'];_require(total<=MAX_RECEIPT_BYTES,'Custody receipt bytes exceed4MiB')
        record['path']=str(path.relative_to(root))
        if path in terminal_paths:
            _require(value.get('status')=='needs_inspection' and value.get('safe_to_release') is False,
                     'Controller or supervisor did not seal the inspected terminal result')
            records.append({**record,'kind':'terminal_result'});continue
        if path==invocation/'invocation.json':
            _require(value.get('root')==str(root) and value.get('invocation_dir')==str(invocation)
                     and value.get('token')==owner.get('token') and value.get('operation')=='execute',
                     'Invocation receipt differs from inspected ownership')
            records.append({**record,'kind':'invocation'});continue
        pid=value.get('pid')
        _require(type(pid) is int and pid>0,'Custody receipt has no known process owner')
        pids.add(pid)
        kind='owner' if path==invocation/'owner.json' else 'handle' if path.name.endswith('-handle.json') else 'exit'
        if kind=='handle':
            monitor=path.name[:-len('-handle.json')]+'-monitor.json'
            _require(value.get('monitor_checkpoint')==monitor,'Handle does not bind its exact monitor receipt')
            handles.append((path.with_name(monitor),pid))
        if kind=='exit':
            _require(value.get('status')=='finished' and type(value.get('exit_code')) is int
                     and value.get('manual_stop_required') is False
                     and (value.get('process_exit_verified') is True or
                          path.name.endswith('-process.json') and 'process_exit_verified' not in value),
                     'Unsealed or unresolved process receipt; no inspection release')
            closed.add(pid);exits[path]=pid
        records.append({**record,'kind':kind,'pid':pid,'exit_code':value.get('exit_code') if kind=='exit' else None})
    _require(pids-{owner_pid}<=closed,'A recorded handle lacks a terminal process receipt')
    _require(all(exits.get(path)==pid for path,pid in handles),'A handle lacks its matching terminal monitor receipt')
    return {'receipts':records,'known_pids':sorted(pids),'receipt_bytes':total,
            'process_check':assert_absent(pids),'bounds':{'inventory_entries':MAX_FILES,
                'receipts':MAX_RECEIPTS,'receipt_bytes':MAX_RECEIPT_BYTES,'inventory_seconds':8,'os_snapshot_seconds':2},
            'scope':'Local source-reuse custody only. No accepted assets, remote-task completion, refunded spend or replay authorization.'}


def recheck(root,proof):
    root=Path(root)
    for record in proof['receipts']:
        path=root/record['path']
        _require(path.is_relative_to(root),'Custody receipt escaped the tool root')
        _,current=_receipt(path)
        _require(current['bytes']==record['bytes'] and current['sha256']==record['sha256'],
                 'Custody receipt changed during inspection')
    return assert_absent(set(proof['known_pids']))
