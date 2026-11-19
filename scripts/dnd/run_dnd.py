#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DND sRGB real-noise filtering benchmark for a multimedia course project.
This program NEVER creates simulated noise and NEVER invents PSNR / SSIM without public GT.
Uses official 1-based MATLAB boxes as observed in established DND evaluation scripts.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

METHODS = [
    ("Baseline", "NO_FILTER", {}),
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

DETAIL_FIELDS = [
    "dataset","sample_id","image_id","patch_id","r0","r1","c0","c1","height","width",
    "method","config","params","elapsed_ms","psnr","ssim","mse", "reference_protocol",
    "flat_hf_ratio","edge_gradient_ratio","mean_abs_change","changed_pixel_pct",
]
SUMMARY_FIELDS = [
    "dataset","method","config","records","avg_psnr_db","avg_ssim","avg_mse","avg_elapsed_ms",
    "avg_flat_hf_ratio","avg_edge_gradient_ratio","avg_mean_abs_change","avg_changed_pixel_pct",
]
ERROR_FIELDS = ["sample_id","image_id","patch_id","error"]
MANIFEST_FIELDS = ["method","config","params","description"]


def output_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fp:
        writer=csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def output_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def manifest(out: Path):
    names={"Baseline":"原始输入不处理","Gaussian":"空间高斯滤波","Median":"各通道中值滤波","Bilateral":"结合空间和颜色权重的双边滤波"}
    output_csv(out/'方法参数定义.csv', MANIFEST_FIELDS,
               [dict(method=m, config=c, params=json.dumps(p,ensure_ascii=False), description=names[m]) for m,c,p in METHODS])


def read_boxes(info, image_id):
    """Official DND boundingboxes are object references in info.mat.
    MATLAB stores [row_start, col_start, row_end, col_end] as 1-based inclusive bounds.
    The official Python protocol crops [start-1:end].
    """
    group=info['info']
    bb=group['boundingboxes']
    # Usually (1, 50), support (50,1) but reject other layouts.
    if bb.shape == (1,50):
        ref=bb[0,image_id-1]
    elif bb.shape == (50,1):
        ref=bb[image_id-1,0]
    else:
        raise ValueError(f'info.boundingboxes 形状不符合官方 (1,50) 或 (50,1): {bb.shape}')
    arr=np.asarray(info[ref]).T
    if arr.shape!=(20,4):
        raise ValueError(f'第 {image_id} 张 boundingboxes 应为 (20,4)，实际 {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'第 {image_id} 张 boundingboxes 含非有限坐标')
    if not np.allclose(arr, np.round(arr), atol=1e-5):
        raise ValueError(f'第 {image_id} 张 boundingboxes 存在非整数坐标')
    return arr.astype(np.int64)


def convert_box(box, height, width):
    # Source convention from MPRNet / DND reference code: [row0-1, row1, col0-1, col1].
    r0=int(box[0]-1); r1=int(box[2]); c0=int(box[1]-1); c1=int(box[3])
    if not (0<=r0<r1<=height and 0<=c0<c1<=width):
        raise ValueError(f'裁剪区域越界: {(r0,r1,c0,c1)}，图片大小 {(height,width)}')
    if (r1-r0)!=(c1-c0) or (r1-r0)!=512:
        raise ValueError(f'官方图块应为 512×512，实际 {(r1-r0,c1-c0)}')
    return r0,r1,c0,c1


def inspect_dataset(root: Path, verify_pixels=False):
    info_path=root/'info.mat'
    srgb=root/'images_srgb'
    if not info_path.is_file():raise FileNotFoundError(f'缺少 info.mat: {info_path}')
    if not srgb.is_dir():raise FileNotFoundError(f'缺少 images_srgb 文件夹: {srgb}')
    index=[]
    problems=[]
    with h5py.File(info_path,'r') as info:
        for image_id in range(1,51):
            imgpath=srgb/f'{image_id:04d}.mat'
            boxes=read_boxes(info,image_id)
            if not imgpath.is_file():
                for patch_id in range(1,21):
                    problems.append(dict(sample_id=f'DND_{image_id:04d}_{patch_id:02d}',image_id=image_id,patch_id=patch_id,error=f'缺失 {imgpath}'))
                continue
            try:
                with h5py.File(imgpath,'r') as f:
                    if 'InoisySRGB' not in f:raise KeyError('缺少 InoisySRGB')
                    ds=f['InoisySRGB']
                    if ds.ndim!=3 or ds.shape[0]!=3:
                        raise ValueError(f'InoisySRGB 实际物理形状 {ds.shape}，预期 (3,W,H)')
                    height,width=int(ds.shape[2]),int(ds.shape[1])
                    for patch_id, box in enumerate(boxes,1):
                        sid=f'DND_{image_id:04d}_{patch_id:02d}'
                        try:
                            r0,r1,c0,c1=convert_box(box,height,width)
                            if verify_pixels:
                                # Decode only requested crop; never load full high-resolution image.
                                x=np.asarray(ds[:,c0:c1,r0:r1])
                                if x.shape!=(3,c1-c0,r1-r0) or not np.all(np.isfinite(x)):
                                    raise ValueError('像素块形状异常或包含 NaN/Inf')
                            index.append(dict(dataset='dnd',sample_id=sid,image_id=image_id,patch_id=patch_id,
                                              r0=r0,r1=r1,c0=c0,c1=c1,height=r1-r0,width=c1-c0,
                                              filename=imgpath.name,verified_pixels=('yes' if verify_pixels else 'not_decoded')))
                        except Exception as exc:
                            problems.append(dict(sample_id=sid,image_id=image_id,patch_id=patch_id,error=str(exc)))
            except Exception as exc:
                for patch_id in range(1,21):
                    sid=f'DND_{image_id:04d}_{patch_id:02d}'
                    problems.append(dict(sample_id=sid,image_id=image_id,patch_id=patch_id,error=f'{imgpath}: {exc}'))
    return index, problems


def load_patch(ds, sample):
    # DND h5py dataset has MATLAB-reversed order (3, width, height).
    # The official reference code obtains HWC using np.array(ds).T.
    # This reads only the desired 512x512 crop, so RAM usage is small.
    x=np.asarray(ds[:,sample['c0']:sample['c1'],sample['r0']:sample['r1']]).transpose(2,1,0)
    if x.ndim!=3 or x.shape[2]!=3 or x.shape[:2]!=(sample['height'],sample['width']):
        raise ValueError(f'图块像素形状错误 {x.shape}')
    if not np.all(np.isfinite(x)):raise ValueError('图块含有 NaN/Inf')
    # Official sRGB source is float in [0,1], not RGB integers.
    mn=float(x.min());mx=float(x.max())
    if mn < -1e-4 or mx > 1.0001:
        raise ValueError(f'InoisySRGB 像素范围异常 [{mn:.5f}, {mx:.5f}]，拒绝随意量化')
    rgb=np.clip(np.rint(x*255),0,255).astype(np.uint8)
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)


def filter_image(img, method, cfg):
    if method=='Baseline':return img.copy()
    if method=='Gaussian':return cv2.GaussianBlur(img, (cfg['k'],cfg['k']),cfg['sigma'])
    if method=='Median':return cv2.medianBlur(img,cfg['k'])
    if method=='Bilateral':return cv2.bilateralFilter(img,cfg['d'],cfg['color'],cfg['space'])
    raise ValueError('unknown filter '+method)


def no_reference_features(noisy, filtered):
    # These are *relative pixel-change proxies*, not ground-truth denoising scores.
    before=cv2.cvtColor(noisy,cv2.COLOR_BGR2GRAY).astype(np.float32)
    after=cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY).astype(np.float32)
    bx=cv2.Sobel(before, cv2.CV_32F,1,0,ksize=3)
    by=cv2.Sobel(before, cv2.CV_32F,0,1,ksize=3)
    grad=cv2.magnitude(bx,by)
    ax=cv2.Sobel(after, cv2.CV_32F,1,0,ksize=3)
    ay=cv2.Sobel(after, cv2.CV_32F,0,1,ksize=3)
    agrad=cv2.magnitude(ax,ay)
    low=float(np.quantile(grad,.25));high=float(np.quantile(grad,.75))
    flat=grad<=low;edges=grad>=high
    bp=cv2.Laplacian(before,cv2.CV_32F,ksize=3)
    ap=cv2.Laplacian(after,cv2.CV_32F,ksize=3)
    hf=float(np.mean(np.abs(ap[flat]))/(np.mean(np.abs(bp[flat]))+1e-6))
    er=float(np.mean(agrad[edges])/(np.mean(grad[edges])+1e-6))
    diff=cv2.absdiff(noisy, filtered)
    mad=float(np.mean(diff))
    changed=float(np.mean(np.any(diff>0,axis=2))*100.0)
    return hf,er,mad,changed


def imgwrite(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok,buff=cv2.imencode('.png',img)
    if not ok: raise RuntimeError(f'图片编码失败: {path}')
    buff.tofile(str(path))


def save_comparison(root, sample, imgs):
    # Original and 3 recommended mid-range settings, with ASCII labels.
    config_ids=['NO_FILTER','G5_s1.1','M5','B7_c35_s5']
    titles=['INPUT - real noise','GAUSSIAN (5x5)','MEDIAN (5x5)','BILATERAL (d7)']
    if any(k not in imgs for k in config_ids):return
    tiles=[]
    for cfg,title in zip(config_ids,titles):
        bgr=imgs[cfg]
        h,w=bgr.shape[:2]
        titlebar=np.full((40,w,3),(26,32,37),dtype=np.uint8)
        cv2.putText(titlebar,title,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.63,(241,246,251),1,cv2.LINE_AA)
        tiles.append(np.concatenate((titlebar,bgr),axis=0))
    combined=np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)],axis=0)
    imgwrite(root/'代表性四图对比'/f"{sample['sample_id']}_四图对比.png",combined)


def load_existing(path):
    if not path.exists():return [],set()
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    keys=set()
    for r in rows:
        k=(r.get('sample_id'),r.get('config'))
        if k in keys:print(f'[WARN] 已有 CSV 存在重复记录：{k}',flush=True)
        keys.add(k)
    return rows,keys


def mean(vals):
    return float(sum(vals)/len(vals)) if vals else None


def summarize(out: Path, expected: int|None=None):
    detail=out/'逐图逐参数结果.csv'
    rows,_=load_existing(detail)
    groups=defaultdict(list);scenes=defaultdict(list)
    for row in rows:
        groups[(row['method'],row['config'])].append(row)
        scenes[(row['image_id'],row['method'],row['config'])].append(row)
    def aggr(key,records):
        m,c=key[-2:]
        def get(col):
            vals=[]
            for r in records:
                raw=r.get(col,'')
                try:
                    if raw not in ('',None):
                        v=float(raw)
                        if math.isfinite(v):vals.append(v)
                except ValueError:pass
            return mean(vals)
        return dict(dataset='dnd',method=m,config=c,records=len(records),
                    avg_psnr_db='',avg_ssim='',avg_mse='',avg_elapsed_ms=get('elapsed_ms'),
                    avg_flat_hf_ratio=get('flat_hf_ratio'),avg_edge_gradient_ratio=get('edge_gradient_ratio'),
                    avg_mean_abs_change=get('mean_abs_change'),avg_changed_pixel_pct=get('changed_pixel_pct'))
    order={cfg:i for i,(_,cfg,_) in enumerate(METHODS)}
    summary=[aggr(k,v) for k,v in groups.items()]
    summary.sort(key=lambda r:order.get(r['config'],999))
    output_csv(out/'按算法参数汇总.csv', SUMMARY_FIELDS,summary)
    sub=[dict(image_id=i,**aggr((m,c),v)) for (i,m,c),v in scenes.items()]
    sub.sort(key=lambda r:(int(r['image_id']),order.get(r['config'],999)))
    output_csv(out/'分图像汇总.csv',['image_id']+SUMMARY_FIELDS,sub)
    ids={r['sample_id'] for r in rows if r['config']=='NO_FILTER'}
    finished=len(rows)
    report={
        'dataset':'dnd','score_type':'NO_REFERENCE','ground_truth_public':False,
        'description':'真实噪声 sRGB 512×512 官方图块；9组滤波+输入基线；无人工加噪',
        'experimental_records':finished,'processed_input_patches':len(ids),
        'expected_records':expected,'complete_selected':finished==expected if expected is not None else None,
        'filters':len(METHODS)-1,'configs_per_patch':len(METHODS),
        'quality_metrics':'PSNR/SSIM/MSE 留空，因为 DND 不公开 GT，不能以带噪图为真值。',
        'flat_hf_ratio':'仅是平坦区域高频相对变化；非噪声真值、非准确率。',
        'edge_gradient_ratio':'仅是输入强边缘区域梯度相对变化；越高不一定越好。',
        'results_file':str(detail),
    }
    output_json(out/'实验完成状态.json',report)
    return report


def run(args):
    data=args.data.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest(out)
    if args.summary_only:
        report=summarize(out)
        print('[SUMMARY]',json.dumps(report,ensure_ascii=False));return 0
    print(f'[DATA] {data}',flush=True)
    print('[VERIFY] 核对 50 张 sRGB MATLAB 图像 / 1000 个官方 bounding boxes ...',flush=True)
    records, problems=inspect_dataset(data, verify_pixels=args.verify_pixels)
    output_csv(out/'官方图块核验.csv',
        ['dataset','sample_id','image_id','patch_id','r0','r1','c0','c1','height','width','filename','verified_pixels'],records)
    output_csv(out/'数据检查错误.csv',ERROR_FIELDS,problems)
    print(f'[VERIFY] 合法图块={len(records)} / 1000，检查错误={len(problems)}；'+('已读取验证每个像素块' if args.verify_pixels else '仅坐标/文件结构检查，尚未完整解码图块'),flush=True)
    if problems:
        for v in problems[:8]: print(f"[WARN] {v['sample_id']}: {v['error']}",flush=True)
    if args.check_only:
        report={'dataset':'dnd','valid_patches':len(records),'expected_patches':1000,
                'errors':len(problems),'pixel_decoding_checked':bool(args.verify_pixels),
                'complete':len(records)==1000 and not problems}
        output_json(out/'检查报告.json',report)
        print('[VERIFY-FINISH]',json.dumps(report,ensure_ascii=False),flush=True)
        return 0 if report['complete'] else 2
    if args.limit < 0:raise ValueError('--limit 不可为负数')
    selected=records[:args.limit] if args.limit else records
    if not selected:raise ValueError('未找到可用图块')
    detail=out/'逐图逐参数结果.csv'
    old,done=load_existing(detail)
    print(f'[START] 官方图块={len(selected)}；配置={len(METHODS)}（基线+9组）；人工加噪=否；PSNR/SSIM=不计算',flush=True)
    print(f'[RESUME] 已有记录={len(old)}，将跳过已完成的组合',flush=True)
    errors=[]
    mode='a' if detail.exists() and detail.stat().st_size>0 else 'w'
    current_image=None
    opened=None
    try:
        with detail.open(mode,encoding='utf-8-sig' if mode=='w' else 'utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=DETAIL_FIELDS)
            if mode=='w':writer.writeheader()
            for j,sample in enumerate(selected,1):
                sid=sample['sample_id']; image_id=sample['image_id'];patch_id=sample['patch_id']
                todo=[(m,c,p) for m,c,p in METHODS if (sid,c) not in done]
                if not todo:
                    if j==len(selected) or j%100==0:print(f'[{j}/{len(selected)}] {sid}：已完成，跳过',flush=True)
                    continue
                try:
                    if image_id!=current_image:
                        if opened is not None:opened.close()
                        opened=h5py.File(data/'images_srgb'/f'{image_id:04d}.mat','r')
                        current_image=image_id
                    img=load_patch(opened['InoisySRGB'],sample)
                    all_results={}
                    for m,c,p in todo:
                        start=time.perf_counter_ns()
                        result=filter_image(img,m,p)
                        elapsed=(time.perf_counter_ns()-start)/1_000_000
                        fh,eg,ma,cp=no_reference_features(img,result)
                        row={k:sample.get(k,'') for k in ('sample_id','image_id','patch_id','r0','r1','c0','c1','height','width')}
                        row.update(dataset='dnd',method=m,config=c,params=json.dumps(p,ensure_ascii=False),
                                   elapsed_ms=round(elapsed,6),psnr='',ssim='',mse='',reference_protocol='no_reference',
                                   flat_hf_ratio=round(fh,6),edge_gradient_ratio=round(eg,6),
                                   mean_abs_change=round(ma,6),changed_pixel_pct=round(cp,6))
                        writer.writerow(row);done.add((sid,c));all_results[c]=result
                        if args.save_all:
                            imgwrite(out/'全部输出图片'/sid/f'{c}.png',result)
                    f.flush()
                    if j<=args.previews:
                        imgwrite(out/'代表性四图对比'/'原始输入图块'/f'{sid}_input.png',img)
                        needed=['NO_FILTER','G5_s1.1','M5','B7_c35_s5']
                        for m,c,p in METHODS:
                            if c in needed and c not in all_results:
                                all_results[c]=filter_image(img,m,p)
                        save_comparison(out,sample,all_results)
                    if j<=5 or j%20==0 or j==len(selected):
                        print(f'[{j}/{len(selected)}] {sid}: 累计记录 {len(done)}',flush=True)
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'
                    print(f'[SKIP] {sid}: {error}',flush=True)
                    errors.append(dict(sample_id=sid,image_id=image_id,patch_id=patch_id,error=error))
    finally:
        if opened is not None:opened.close()
    output_csv(out/'运行错误记录.csv',ERROR_FIELDS,errors)
    target=len(selected)*len(METHODS)
    status=summarize(out,expected=target)
    status['missing_from_check']=len(problems)
    status['execution_errors']=len(errors)
    status['selected_patches']=len(selected)
    status['complete_selected']=status['complete_selected'] and not errors and not problems
    output_json(out/'实验完成状态.json',status)
    print('[FINISH]',json.dumps({k:status[k] for k in ('dataset','selected_patches','expected_records','experimental_records','complete_selected','execution_errors')},ensure_ascii=False),flush=True)
    print(f'[OUTPUT] {out}',flush=True)
    return 0 if status['complete_selected'] else 2


def main():
    p=argparse.ArgumentParser(description='DND 真实噪声 sRGB 数据集：官方 1000 图块、9 组滤波、无人工加噪、无 GT 不评分')
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'DND',help='包含 images_srgb 和 info.mat 的目录')
    p.add_argument('--out',type=Path,required=True,help='本次实验专用输出目录；不要混用测试和全量目录')
    p.add_argument('--limit',type=int,default=0,help='测试图块数量，0=全部')
    p.add_argument('--previews',type=int,default=8,help='仅保存前 N 个四图对比预览，默认 8')
    p.add_argument('--check-only',action='store_true',help='只检查官方图块坐标/文件结构，不进行滤波')
    p.add_argument('--verify-pixels',action='store_true',help='逐图块解码核验像素，会较慢')
    p.add_argument('--summary-only',action='store_true',help='不处理图像，仅从现有明细重建统计汇总')
    p.add_argument('--save-all',action='store_true',help='可选：保存每组滤波后的全部 PNG，非常占磁盘')
    args=p.parse_args()
    try:return run(args)
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 2

if __name__=='__main__':sys.exit(main())
