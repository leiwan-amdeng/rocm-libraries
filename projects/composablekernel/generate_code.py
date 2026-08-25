#!/usr/bin/env python3
"""
Regenerate full inline source-map files for a rocprofv3 ATT profile.

Default behavior is SAFE for rocprof-compute-viewer flamegraph: keep the original
ui_output code.json statistics columns (Hit/Latency/Stall/Idle) and only merge
full inline Source strings generated from the current profile's code object.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

DEFAULT_SNAPSHOT_SCRIPT = Path("/nfs_share/disk2/wanglei/rocprof-compute-viewer/scripts/generate_snapshot.py")

def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(1)

def pick_one(paths: list[Path], desc: str) -> Path:
    if not paths: fail(f"no {desc} found")
    return sorted(paths)[-1]

def inspect_code_json(path: Path):
    data=json.loads(path.read_text()); rows=data if isinstance(data,list) else data.get('code',[])
    lens=[]; hit=lat=stall=idle=0
    for r in rows:
        src=''
        if isinstance(r,list):
            if len(r)>=4: src=str(r[3] or '')
            if len(r)>=10:
                try:
                    hit+=int(r[6] or 0); lat+=int(r[7] or 0); stall+=int(r[8] or 0); idle+=int(r[9] or 0)
                except Exception: pass
        elif isinstance(r,dict): src=str(r.get('Source') or r.get('source') or '')
        if src: lens.append(len(src))
    return len(rows), len(lens), (sum(lens)/len(lens) if lens else 0.0), (max(lens) if lens else 0), hit, lat, stall, idle

def find_ui_dir(profile: Path, explicit: str|None) -> Path:
    if explicit:
        ui=Path(explicit).resolve()
        if not ui.is_dir(): fail(f"explicit ui dir does not exist: {ui}")
        return ui
    ui_dirs=[p for p in profile.glob('ui_output_agent_*_dispatch_*') if p.is_dir() and '_backup_' not in p.name]
    if not ui_dirs: fail('no ui_output_agent_*_dispatch_* directory')
    def score(p: Path):
        size=(p/'code.json').stat().st_size if (p/'code.json').exists() else 0
        try: disp=int(p.name.rsplit('_dispatch_',1)[1])
        except Exception: disp=-1
        return (size, disp, str(p))
    return sorted(ui_dirs, key=score)[-1]

def restore_latest_backup(profile: Path, ui: Path) -> None:
    backups=[p for p in profile.glob(ui.name+'_short_codejson_backup_*') if p.is_dir()]
    if not backups: fail(f"no backup dirs found for {ui.name}")
    bak=sorted(backups)[-1]
    print(f"[restore] restoring UI files from latest backup: {bak}")
    for f in ui.glob('source_*'):
        if f.is_file(): f.unlink()
    for name in ('code.json','snapshots.json'):
        src=bak/name
        if src.exists(): shutil.copy2(src, ui/name)
    for src in bak.glob('source_*'):
        if src.is_file(): shutil.copy2(src, ui/src.name)
    print('[restore] done')

def source_map_from_full(full_code: Path) -> dict[tuple[int,int], str]:
    data=json.loads(full_code.read_text()); rows=data if isinstance(data,list) else data.get('code',[])
    m={}
    for r in rows:
        if not isinstance(r,list) or len(r)<6: continue
        src=str(r[3] or '')
        if not src: continue
        try: key=(int(r[4]), int(r[5]))
        except Exception: continue
        old=m.get(key)
        if old is None or len(src)>len(old): m[key]=src
    return m

def merge_source_only(ui_code: Path, full_code: Path) -> int:
    ui_data=json.loads(ui_code.read_text())
    if not isinstance(ui_data,dict) or 'code' not in ui_data: fail(f'unexpected UI code.json schema: {ui_code}')
    fmap=source_map_from_full(full_code); changed=0
    for r in ui_data.get('code',[]):
        if not isinstance(r,list) or len(r)<6: continue
        try: key=(int(r[4]), int(r[5]))
        except Exception: continue
        src=fmap.get(key)
        if src and len(src)>len(str(r[3] or '')):
            r[3]=src; changed+=1
    ui_code.write_text(json.dumps(ui_data, ensure_ascii=False))
    return changed

def main():
    ap=argparse.ArgumentParser(description='Generate/merge full inline source map for rocprofv3 ATT profile.')
    ap.add_argument('profile_dir')
    ap.add_argument('--snapshot-script', default=str(DEFAULT_SNAPSHOT_SCRIPT))
    ap.add_argument('--codeobj')
    ap.add_argument('--no-replace', action='store_true')
    ap.add_argument('--ui-dir')
    ap.add_argument('--replace-whole-code-json', action='store_true', help='DANGEROUS: replace entire ui code.json; loses stats if fullsrc has zeros')
    ap.add_argument('--restore-backup', action='store_true')
    args=ap.parse_args()
    profile=Path(args.profile_dir).resolve()
    if not profile.is_dir(): fail(f'profile_dir does not exist: {profile}')
    ui=find_ui_dir(profile,args.ui_dir)
    if args.restore_backup:
        restore_latest_backup(profile,ui)
        if args.no_replace: return
    snapshot_script=Path(args.snapshot_script)
    if not snapshot_script.is_file(): fail(f'generate_snapshot.py not found: {snapshot_script}')
    if args.codeobj:
        codeobj=Path(args.codeobj).resolve()
        if not codeobj.is_file(): fail(f'explicit code object does not exist: {codeobj}')
    else:
        codeobj=pick_one(list(profile.glob('*_gfx1201_code_object_id_2.out')) or list(profile.glob('*_gfx1201_code_object_id_*.out')), 'gfx1201 code object')
    full_dir=profile.with_name(profile.name+'_fullsrc')
    if full_dir.exists(): shutil.rmtree(full_dir)
    full_dir.mkdir(parents=True)
    print(f'[1/3] profile: {profile}'); print(f'[1/3] ui     : {ui}'); print(f'[1/3] codeobj: {codeobj}'); print(f'[1/3] fullsrc: {full_dir}')
    for att in profile.glob('*.att'): shutil.copy2(att, full_dir/att.name)
    shutil.copy2(codeobj, full_dir/codeobj.name)
    print('[2/3] running generate_snapshot.py ...')
    subprocess.run([sys.executable, str(snapshot_script), codeobj.name], cwd=str(full_dir), check=True)
    full_code=full_dir/'code.json'; full_snap=full_dir/'snapshots.json'
    if not full_code.is_file() or full_code.stat().st_size==0: fail(f'full code.json not generated: {full_code}')
    if not full_snap.is_file() or full_snap.stat().st_size==0: fail(f'snapshots.json not generated: {full_snap}')
    rows,nonempty,avg,mx,hit,lat,stall,idle=inspect_code_json(full_code)
    print(f'[check] full code.json rows={rows} nonempty_source={nonempty} avg_source_len={avg:.1f} max_source_len={mx} hit={hit} latency={lat}')
    print(f'[check] full source_* count={len(list(full_dir.glob("source_*")))}')
    if args.no_replace:
        print('[3/3] --no-replace set; not touching ui_output directory'); print(f'DONE full source map: {full_dir}'); return
    ts=time.strftime('%Y%m%d_%H%M%S'); bak=ui.with_name(ui.name+f'_short_codejson_backup_{ts}'); bak.mkdir(parents=True, exist_ok=False)
    print(f'[3/3] updating UI source map in: {ui}'); print(f'[3/3] backup old files to     : {bak}')
    for name in ('code.json','snapshots.json'):
        src=ui/name
        if src.exists(): shutil.copy2(src,bak/name)
    for src in ui.glob('source_*'):
        if src.is_file(): shutil.copy2(src,bak/src.name)
    if args.replace_whole_code_json:
        print('[WARN] replacing whole code.json; this can zero flamegraph stats')
        shutil.copy2(full_code, ui/'code.json')
    else:
        changed=merge_source_only(ui/'code.json', full_code)
        print(f'[3/3] merged full Source strings into existing UI code.json rows: changed={changed}')
    shutil.copy2(full_snap, ui/'snapshots.json')
    for src in ui.glob('source_*'):
        if src.is_file(): src.unlink()
    for src in full_dir.glob('source_*'):
        if src.is_file(): shutil.copy2(src, ui/src.name)
    rows,nonempty,avg,mx,hit,lat,stall,idle=inspect_code_json(ui/'code.json')
    print(f'[check] UI code.json rows={rows} nonempty_source={nonempty} avg_source_len={avg:.1f} max_source_len={mx}')
    print(f'[check] UI stats totals: hit={hit} latency={lat} stall={stall} idle={idle}')
    print(f'[check] UI source_* count={len(list(ui.glob("source_*")))}')
    print('DONE'); print(f'PROFILE_DIR={profile}'); print(f'FULL_DIR={full_dir}'); print(f'UI_DIR={ui}'); print(f'BACKUP_DIR={bak}')
if __name__=='__main__': main()
