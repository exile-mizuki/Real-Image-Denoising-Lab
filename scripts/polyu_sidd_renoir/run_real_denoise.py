# -*- coding: utf-8 -*-
"""真实自然图像去噪实验：PolyU / SIDD / RENOIR。
不人工加噪；保留原图像素尺度；配对时绝不自动缩放 GT。
独立实现，仅参考开源项目的实验组织方式。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from skimage.metrics import structural_similarity

CONFIGS = [
    ("Gaussian", "G3_s0.8", {"k":3,"sigma":0.8}),
    ("Gaussian", "G5_s1.1", {"k":5,"sigma":1.1}),
    ("Gaussian", "G7_s1.5", {"k":7,"sigma":1.5}),
    ("Median", "M3", {"k":3}),
    ("Median", "M5", {"k":5}),
    ("Median", "M7", {"k":7}),
    ("Bilateral", "B5_c20_s3", {"d":5,"color":20,"space":3}),
    ("Bilateral", "B7_c35_s5", {"d":7,"color":35,"space":5}),
    ("Bilateral", "B9_c50_s7", {"d":9,"color":50,"space":7}),
]
FIELDNAMES = ["dataset", "pair_id", "input_file", "reference_file", "crop_xywh", "width", "height", "method", "config", "psnr_db", "ssim", "mse", "elapsed_ms", "metric_note"]

@dataclass
class Pair:
    key: str
    noisy: Path
    gt: Path | None
    group: str = ""


def walk_find_dir(root: Path, folder: str) -> list[Path]:
    if root.name.casefold() == folder.casefold():
        return [root]
    return sorted((d for d in root.rglob("*") if d.is_dir() and d.name.casefold() == folder.casefold()), key=str)


def polyu_pairs(root: Path) -> list[Pair]:
    for d in walk_find_dir(root, "train_sub"):
        gt = d / "GT"
        lq = d / "LQ"
        if gt.is_dir() and lq.is_dir():
            gt_files = {p.name.casefold():p for p in gt.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
            lq_files = {p.name.casefold():p for p in lq.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
            missing = sorted(set(lq_files) - set(gt_files))
            unused = sorted(set(gt_files) - set(lq_files))
            print(f"[PolyU] GT={len(gt_files)}, LQ={len(lq_files)}, 配对={len(gt_files.keys() & lq_files.keys())}, 缺GT={len(missing)}, 多余GT={len(unused)}", flush=True)
            if missing: print("  缺GT示例:", missing[:5])
            if unused: print("  多余GT示例:", unused[:5])
            return [Pair(lq_files[k].stem, lq_files[k], gt_files[k]) for k in sorted(gt_files.keys() & lq_files.keys())]
    raise FileNotFoundError("找不到 train_sub/GT 和 train_sub/LQ。请将 --data 指向解压的 PolyU 文件夹。")


def sidd_pairs(root: Path) -> list[Pair]:
    """Discover ALL NNNN_GT_SRGB / NNNN_NOISY_SRGB sibling folders.

    The four-digit scene ID stays part of the pair ID. Never match images
    merely because they share their 001.PNG suffix in different scenes.
    """
    result = []
    gt_pattern = re.compile(r"^(\d{4})_GT_SRGB$", re.I)
    gt_dirs = [p for p in [root, *root.rglob("*")] if p.is_dir() and gt_pattern.fullmatch(p.name)]
    seen = set()
    for gt_dir in sorted(gt_dirs, key=str):
        scene = gt_pattern.fullmatch(gt_dir.name).group(1)
        noisy_dir = gt_dir.parent / f"{scene}_NOISY_SRGB"
        if not noisy_dir.is_dir():
            # Case-insensitive path matching for cross-platform consistency.
            matches = [p for p in gt_dir.parent.iterdir() if p.is_dir() and p.name.casefold() == f"{scene}_NOISY_SRGB".casefold()]
            noisy_dir = matches[0] if matches else noisy_dir
        if not noisy_dir.is_dir():
            print(f"[SIDD][WARN] {scene} 缺少 NOISY 目录：{noisy_dir}", flush=True)
            continue
        def keyed(directory, mode):
            result_map = {}
            # SIDD 逐场景 ZIP 的图片可能是 GT_SRGB_001.PNG / NOISY_SRGB_001.PNG，
            # 也可能是 0035_GT_SRGB_001.PNG / 0035_NOISY_SRGB_001.PNG。
            # 不能把 GT_SRGB 与 NOISY_SRGB 保留在配对键中，否则 150+150 会配对为 0。
            pattern = re.compile(rf"^(?:{re.escape(scene)}_)?{re.escape(mode)}_SRGB_(.+)$", re.I)
            unmatched = []
            for image_file in directory.iterdir():
                if not image_file.is_file() or image_file.suffix.casefold() not in (".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                match = pattern.fullmatch(image_file.stem)
                if not match:
                    unmatched.append(image_file.name)
                    continue
                suffix = match.group(1).casefold()
                if suffix in result_map:
                    raise ValueError(f"SIDD 同一场景内部配对编号重复：{scene} {suffix}")
                result_map[suffix] = image_file
            if unmatched:
                print(f"[SIDD][WARN] {scene} {mode} 有 {len(unmatched)} 张命名不符合预期，已跳过；示例：{unmatched[:3]}", flush=True)
            return result_map
        a = keyed(gt_dir, "GT")
        b = keyed(noisy_dir, "NOISY")
        keys = sorted(set(a) & set(b))
        print(f"[SIDD][{scene}] GT={len(a)}, NOISY={len(b)}, 配对={len(keys)}；缺GT={len(set(b)-set(a))}；缺NOISY={len(set(a)-set(b))}", flush=True)
        for suffix in keys:
            key = f"SIDD_{scene}_{suffix}"
            if key in seen:
                raise ValueError(f"SIDD 配对编号重复：{key}；请移走重复解压的文件夹")
            seen.add(key)
            result.append(Pair(key, b[suffix], a[suffix], scene))
    if not result:
        raise FileNotFoundError("找不到任何 NNNN_GT_SRGB 与相邻 NNNN_NOISY_SRGB 成对的图像；请检查 --data 路径和解压目录。")
    return result


def renoir_pairs(root: Path) -> list[Pair]:
    """Discover multiple cameras; camera name is part of the unique pair ID.

    References are for preview only by default, NEVER PSNR/SSIM scoring.
    """
    result = []
    batch_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.casefold().startswith("batch_")]
    seen = set()
    allowed_ext = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for batch in sorted(batch_dirs, key=str):
        parts = [p.casefold() for p in batch.parts]
        camera = next((part.split("_")[0] for part in batch.parts if part.casefold() in {"s90_aligned", "mi3_aligned", "t3i_aligned"}), None)
        if camera is None and root.name.casefold() in {"s90_aligned", "mi3_aligned", "t3i_aligned"}:
            camera = root.name.split("_")[0]
        if camera is None:
            print(f"[RENOIR][WARN] 略过无法识别相机的目录：{batch}", flush=True)
            continue
        camera = {"s90": "S90", "mi3": "Mi3", "t3i": "T3i"}[camera.casefold()]
        images = [p for p in batch.iterdir() if p.is_file() and p.suffix.casefold() in allowed_ext]
        refs = [p for p in images if "reference" in p.stem.casefold() and "mask" not in p.stem.casefold()]
        noisy = [p for p in images if "noisy" in p.stem.casefold() and "plot" not in p.stem.casefold() and "mask" not in p.stem.casefold()]
        if len(refs) != 1:
            print(f"[RENOIR][WARN] 跳过 {camera}/{batch.name}：Reference 数量={len(refs)}（需恰好1张）", flush=True)
            continue
        for p in sorted(noisy, key=str):
            key = f"RENOIR_{camera}_{batch.name}_{p.stem}"
            if key in seen:
                raise ValueError(f"RENOIR 配对编号重复：{key}")
            seen.add(key)
            result.append(Pair(key, p, refs[0], camera))
    if not result:
        raise FileNotFoundError("找不到相机目录下 Batch_*/...Reference 与 ...Noisy 图像组合；请检查目录结构和文件扩展名。")
    from collections import Counter
    counts = Counter(p.group for p in result)
    print(f"[RENOIR] 共 {len(result)} 对候选样本；相机统计={dict(sorted(counts.items()))}。默认仅做视觉实验，不把 Reference 当作已校验严格配准的 GT。", flush=True)
    return result


def read_bgr(path:Path) -> np.ndarray:
    raw=np.fromfile(str(path),dtype=np.uint8) # 支持中文 Windows 路径
    img=cv2.imdecode(raw,cv2.IMREAD_COLOR)
    if img is None:raise ValueError(f"无法读取图片：{path}")
    return img


def save_png(path:Path, img:np.ndarray):
    path.parent.mkdir(parents=True,exist_ok=True)
    okay,data=cv2.imencode(".png",img)
    if not okay:raise IOError(f"PNG 编码失败：{path}")
    data.tofile(str(path))


def patch_pair(input_img:np.ndarray, reference:np.ndarray|None, size:int, key:str):
    h,w=input_img.shape[:2]
    if reference is not None and input_img.shape!=reference.shape:
        raise ValueError(f"输入与参考图尺寸不同：{input_img.shape} vs {reference.shape}；禁止强行缩放后评分")
    if size<=0:
        return input_img, reference, (0,0,w,h)
    ph=min(size,h);pw=min(size,w)
    seed=int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8],16)
    x=seed%(w-pw+1)
    y=(seed//4096)%(h-ph+1)
    a=input_img[y:y+ph,x:x+pw]
    b=reference[y:y+ph,x:x+pw] if reference is not None else None
    return a,b,(x,y,pw,ph)


def run_filter(noisy,method, params):
    if method=="Gaussian":return cv2.GaussianBlur(noisy,(params["k"],params["k"]),params["sigma"])
    if method=="Median":return cv2.medianBlur(noisy,params["k"])
    if method=="Bilateral":return cv2.bilateralFilter(noisy,params["d"],params["color"],params["space"])
    raise ValueError(method)


def metrics(gt, out):
    if gt is None:return ("", "", "")
    d=gt.astype(np.float32)-out.astype(np.float32)
    mse=float(np.mean(d*d))
    psnr=10*math.log10(255*255/mse) if mse>0 else float("inf")
    ssim=float(structural_similarity(gt,out,data_range=255,channel_axis=2)) if min(gt.shape[:2])>=7 else float('nan')
    return (f"{psnr:.6f}",f"{ssim:.6f}",f"{mse:.6f}")


def labeled(img,label):
    h,w=img.shape[:2]
    strip=np.full((32,w,3),242,np.uint8)
    cv2.putText(strip,label,(8,23),cv2.FONT_HERSHEY_SIMPLEX,0.57,(20,20,20),1,cv2.LINE_AA)
    return np.vstack([strip,img])


def make_preview(pair, noisy, gt, outs, preview_dir):
    panels=[labeled(noisy,"Noisy input")]
    for key in ("G5_s1.1","M3","B7_c35_s5"):
        if key in outs:panels.append(labeled(outs[key],key))
    if gt is not None:panels.append(labeled(gt,"Reference (unverified)" if pair.key.startswith("RENOIR_") else "Reference GT"))
    img=cv2.hconcat(panels)
    safe=re.sub(r"[^0-9A-Za-z_-]+","_",pair.key)[:80]
    save_png(preview_dir/(safe+"_comparison.png"),img)


def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def summaries(rows,out):
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r["method"],r["config"])].append(r)
    agg=[]
    for (method,config),members in sorted(grouped.items()):
        v={"method":method,"config":config,"records":len(members)}
        for k in ("psnr_db","ssim","mse","elapsed_ms"):
            vals=[]
            for r in members:
                try:
                    x=float(r[k])
                    if math.isfinite(x):vals.append(x)
                except (ValueError,TypeError,KeyError):pass
            v["avg_"+k]=f"{mean(vals):.6f}" if vals else ""
        agg.append(v)
    write_csv(out/"按算法参数汇总.csv",agg,["method","config","records","avg_psnr_db","avg_ssim","avg_mse","avg_elapsed_ms"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for val,ylab,fn in [("avg_psnr_db","PSNR (dB)","平均PSNR.png"),("avg_ssim","SSIM","平均SSIM.png"),("avg_elapsed_ms","Milliseconds / patch","平均耗时.png")]:
            xs=[x["config"] for x in agg if x[val] != ""]
            ys=[float(x[val]) for x in agg if x[val]!=""]
            if not xs:continue
            fig,ax=plt.subplots(figsize=(max(8,len(xs)*0.85),4.3))
            ax.bar(xs,ys)
            ax.set_ylabel(ylab)
            ax.tick_params(axis="x",rotation=35)
            fig.tight_layout()
            fig.savefig(out/fn,dpi=150)
            plt.close(fig)
    except ImportError:
        print("[NOTICE] 未安装 matplotlib，跳过柱状图；CSV 正常输出。")
    return agg


def run(args):
    root=Path(args.data).resolve()
    if not root.is_dir():raise FileNotFoundError(f"找不到数据目录：{root}")
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    pair_fn={"polyu":polyu_pairs,"sidd":sidd_pairs,"renoir":renoir_pairs}[args.dataset]
    pairs=pair_fn(root)
    requested = set()
    if args.dataset == "sidd" and args.sidd_scenes:
        requested = {x.strip() for x in args.sidd_scenes.split(",") if x.strip()}
        if not all(re.fullmatch(r"\d{4}", x) for x in requested):
            raise ValueError("--sidd-scenes 只能输入四位编号，例如 0035,0063")
    elif args.dataset == "renoir" and args.renoir_cameras:
        canon = {"s90": "S90", "mi3": "Mi3", "t3i": "T3i"}
        items = [x.strip().casefold() for x in args.renoir_cameras.split(",") if x.strip()]
        if not items or any(x not in canon for x in items):
            raise ValueError("--renoir-cameras 只能包含 S90,Mi3,T3i")
        requested = {canon[x] for x in items}
    if requested:
        available = {p.group for p in pairs}
        missing = requested - available
        if missing:
            raise FileNotFoundError(f"请求的数据组未找到有效配对：{sorted(missing)}；当前实际发现：{sorted(available)}")
        pairs = [p for p in pairs if p.group in requested]
    if args.per_group:
        chosen=[]
        group_counts={}
        for p in pairs:
            count=group_counts.get(p.group,0)
            if count<args.per_group:
                chosen.append(p)
                group_counts[p.group]=count+1
        pairs=chosen
    if args.limit:pairs=pairs[:args.limit]
    if not pairs:raise ValueError("没有符合筛选条件的有效配对，请检查数据目录")
    print(f"[START] 数据集={args.dataset} 待处理配对={len(pairs)}；裁剪={args.crop or '整幅'}；算法=3种/9组；人工加噪=否",flush=True)
    pair_rows=[]
    for p in pairs:
        pair_rows.append({"pair_id":p.key,"LQ_or_NOISY":str(p.noisy),"GT_or_REFERENCE":str(p.gt) if p.gt else ""})
    write_csv(out/"配对清单.csv",pair_rows,["pair_id","LQ_or_NOISY","GT_or_REFERENCE"])
    if args.check_only:
        # PIL 只读图片头，快速核对图片尺寸，避免全部载入内存。
        from PIL import Image
        checked=[];bad=0
        for pair in pairs:
            state="OK";dims1="";dims2=""
            try:
                with Image.open(pair.noisy) as img:dims1=f"{img.width}x{img.height}"
                if pair.gt:
                    with Image.open(pair.gt) as img:dims2=f"{img.width}x{img.height}"
                if pair.gt and dims1!=dims2:state="SHAPE_MISMATCH"
            except (OSError,ValueError) as e:
                state="READ_ERROR: "+str(e)
            if state!="OK":bad+=1
            checked.append({"pair_id":pair.key,"input_size":dims1,"reference_size":dims2,"status":state})
        write_csv(out/"配对尺寸核查.csv",checked,["pair_id","input_size","reference_size","status"])
        print(f"[VERIFY] 发现候选配对 {len(pairs)}，尺寸/可读性有问题 {bad}；查看 配对尺寸核查.csv",flush=True)
        return 0 if bad==0 else 2

    config={"dataset":args.dataset,"data":str(root),"crop":args.crop,"algorithm_parameters":CONFIGS,"renoir_score":args.renoir_score,
            "sidd_scenes":args.sidd_scenes,"renoir_cameras":args.renoir_cameras,"per_group":args.per_group,"limit":args.limit}
    confpath=out/"实验设置.json"
    # CONFIGS contains tuples; JSON round-trip converts them to lists.
    # Compare normalized representations so interrupted experiments can resume.
    normalized_config=json.loads(json.dumps(config,ensure_ascii=False))
    if confpath.exists() and json.loads(confpath.read_text(encoding="utf-8"))!=normalized_config:
        raise RuntimeError("输出目录中存在不同参数的实验，请更换 --out 路径，避免混合结果。")
    confpath.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    resultpath=out/"逐图逐参数结果.csv"
    finished=set();existing=[]
    if resultpath.exists():
        with resultpath.open("r",encoding="utf-8-sig",newline="") as f:
            existing=list(csv.DictReader(f))
        finished={(r["pair_id"],r["config"]) for r in existing}
        print(f"[RESUME] 已完成记录 {len(finished)}，跳过重复组合")

    # Existing CSV is UTF-8 with BOM in its header; append without writing a second BOM.
    csv_encoding = "utf-8-sig" if not resultpath.exists() or resultpath.stat().st_size==0 else "utf-8"
    with resultpath.open("a",encoding=csv_encoding,newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDNAMES)
        if resultpath.stat().st_size==0:w.writeheader()
        for index,pair in enumerate(pairs,1):
            try:
                noisy=read_bgr(pair.noisy)
                gt=read_bgr(pair.gt) if pair.gt else None
                if args.dataset=="renoir" and not args.renoir_score and gt is not None and gt.shape!=noisy.shape:
                    print(f"[RENOIR][WARN] {pair.key}：Reference 与 Noisy 尺寸不一致，仅处理 Noisy，不显示参考图、不评分",flush=True)
                    gt=None
                noisy,gt,xywh=patch_pair(noisy,gt,args.crop,pair.key)
            except (ValueError,OSError,cv2.error) as e:
                print(f"[SKIP] {pair.key}: {e}",flush=True)
                continue
            use_gt=gt if args.dataset!="renoir" or args.renoir_score else None
            note=("人工加噪=否；RENOIR默认不评分，需人工核实配准和掩膜" if args.dataset=="renoir" and not args.renoir_score else "人工加噪=否；与配对参考图逐像素比较")
            preview_outputs={}
            for method,config_id,params in [("Baseline","NO_FILTER",{})]+CONFIGS:
                if (pair.key,config_id) in finished and not args.save_all:continue
                start=time.perf_counter()
                output=noisy if method=="Baseline" else run_filter(noisy,method,params)
                elapsed=(time.perf_counter()-start)*1000
                if config_id in ("G5_s1.1","M3","B7_c35_s5"):
                    preview_outputs[config_id]=output
                pval,sval,mval=metrics(use_gt,output)
                row={"dataset":args.dataset,"pair_id":pair.key,"input_file":str(pair.noisy),"reference_file":str(pair.gt) if pair.gt else "", "crop_xywh":"x=%d,y=%d,w=%d,h=%d"%xywh,"width":xywh[2],"height":xywh[3],"method":method,"config":config_id,"psnr_db":pval,"ssim":sval,"mse":mval,"elapsed_ms":f"{elapsed:.3f}","metric_note":note}
                if (pair.key,config_id) not in finished:
                    w.writerow(row);f.flush();finished.add((pair.key,config_id))
                if args.save_all:
                    safe=re.sub(r"[^0-9A-Za-z_-]+","_",pair.key)[:75]
                    save_png(out/"全部处理图"/safe/(config_id+".png"),output)
            if index<=args.previews and preview_outputs:
                make_preview(pair,noisy,gt,preview_outputs,out/"四图或五图对比")
            if index<=5 or index%20==0 or index==len(pairs):
                print(f"[{index}/{len(pairs)}] {pair.key}: 进度正常；累计记录 {len(finished)}",flush=True)
    with resultpath.open("r",encoding="utf-8-sig",newline="") as f:all_rows=list(csv.DictReader(f))
    agg=summaries(all_rows,out)
    expected=len(pairs)*(1+len(CONFIGS))
    done=sum((p.key,c) in finished for p in pairs for c in ["NO_FILTER"]+[m[1] for m in CONFIGS])
    status={"dataset":args.dataset,"pairs_selected":len(pairs),"configs_per_pair":10,"expected_records":expected,"finished_selected_records":done,"complete_selected":done==expected,"pair_errors":expected-done,"scoreless_renoir":args.dataset=="renoir" and not args.renoir_score,"results_path":str(resultpath)}
    (out/"完成状态.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    print("[FINISH]",json.dumps(status,ensure_ascii=False),flush=True)
    if agg:
        scored=[a for a in agg if a["method"]!="Baseline" and a["avg_psnr_db"]]
        if scored:
            best=max(scored,key=lambda a:float(a["avg_psnr_db"]))
            print(f"[REFERENCE] 本次样本按平均PSNR最高: {best['config']} ({best['avg_psnr_db']}dB)；须结合SSIM、图像质量解释")
    return 0 if status["complete_selected"] else 2


def main():
    parser=argparse.ArgumentParser(description="真实图像去噪：PolyU / SIDD / RENOIR，无人工加噪")
    parser.add_argument("--dataset",required=True,type=str.lower,choices=["polyu","sidd","renoir"])
    parser.add_argument("--data",required=True,help="数据集文件夹，支持上层目录自动定位")
    parser.add_argument("--out",required=True,help="独立的结果文件夹")
    parser.add_argument("--limit",type=int,default=0,help="取前N对测试；0全部")
    parser.add_argument("--crop",type=int,default=512,help="原生分辨率裁剪边长，0=整幅原图，无缩放")
    parser.add_argument("--check-only",action="store_true",help="只检查文件名配对并导出清单，不进行滤波")
    parser.add_argument("--renoir-score",action="store_true",help="配准/掩膜核查后才建议用；否则RENOIR默认视觉实验")
    parser.add_argument("--save-all",action="store_true",help="保存每张图的全部9种输出，额外占用较多磁盘")
    parser.add_argument("--previews",type=int,default=6,help="保存多少张代表性拼图")
    parser.add_argument("--sidd-scenes",default="",help="仅处理这些SIDD场景，例如 0035,0063,0094,0118,0146")
    parser.add_argument("--renoir-cameras",default="",help="仅处理这些RENOIR设备，例如 Mi3,T3i")
    parser.add_argument("--per-group",type=int,default=0,help="每个场景/相机测试多少对，0表示不限制")
    a=parser.parse_args()
    try:return run(a)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}",file=sys.stderr)
        return 2

if __name__=="__main__":sys.exit(main())
