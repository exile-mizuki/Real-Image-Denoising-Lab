"""NIND 126 scene / 836 original image manifest: deterministic native-resolution patch experiments.

Run: python scripts/run_full_nind.py --root 原始NIND --out 全量实验结果

No synthetic noise, no training. 3 classical filters x 3 parameter settings on
4 deterministic, native-resolution 256x256 patches per non-reference ISO image.
True PSNR/SSIM vs low-ISO photograph are withheld on misaligned pairs.
All work is incremental and recoverable after interruption.
"""
import argparse,csv,json,math,pathlib,time,sys,hashlib,collections,os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from PIL import Image,ImageDraw,ImageFont
from nind_manifest import records,filename,all_files

PSET=[('Gaussian','k3_s0.8',{'k':3,'sigma':0.8}),('Gaussian','k5_s1.2',{'k':5,'sigma':1.2}),
      ('Gaussian','k7_s1.8',{'k':7,'sigma':1.8}),('Median','k3',{'k':3}),('Median','k5',{'k':5}),
      ('Median','k7',{'k':7}),('Bilateral','d5_c20_s3',{'d':5,'color':20,'space':3}),
      ('Bilateral','d7_c40_s5',{'d':7,'color':40,'space':5}),('Bilateral','d9_c70_s7',{'d':9,'color':70,'space':7})]
FIELDS=['family','scene','reference_iso','iso','input_path','reference_path','patch','x','y','patch_size',
        'source_bits','pair_status','shift_x_native_px','shift_y_native_px','phase_response','method','params','mse_to_lowISO',
        'psnr_to_lowISO','ssim_to_lowISO','psnr_gain_over_input','flat_hf_ratio','edge_gradient_ratio',
        'mean_abs_change','run_ms','note']

def filter_rgb(a,method,pa):
    if method=='Gaussian':return cv2.GaussianBlur(a,(pa['k'],pa['k']),pa['sigma'])
    if method=='Median':return cv2.medianBlur(a,pa['k'])
    if method=='Bilateral':return cv2.bilateralFilter(a,pa['d'],pa['color'],pa['space'])
    if method=='Input':return a
    raise KeyError(method)

def read_rgb_u8(p):
    # cv2 is used rather than PIL.convert('RGB') because some NIND PNGs are 16-bit RGB.
    arr=cv2.imdecode(np.fromfile(str(p),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if arr is None:raise ValueError(f'corrupt/unreadable: {p}')
    if arr.ndim==2:arr=cv2.cvtColor(arr,cv2.COLOR_GRAY2RGB)
    elif arr.shape[2]==4:arr=cv2.cvtColor(arr,cv2.COLOR_BGRA2RGB)
    elif arr.shape[2]==3:arr=cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)
    else:raise ValueError('unsupported shape '+str(arr.shape))
    if arr.dtype==np.uint16:return np.clip(np.rint(arr.astype(np.float32)/257.0),0,255).astype(np.uint8),16
    if arr.dtype==np.uint8:return arr,8
    raise ValueError('unsupported dtype '+str(arr.dtype))

def normalized_patch_positions(h,w,side,count):
    if min(h,w)<side:return []
    # deterministic grid locations, repeated centers removed for small inputs
    n=math.ceil(math.sqrt(count));pos=[]
    for yi in range(n):
        for xi in range(n):
            x=int(round((w-side)*(xi+.5)/n));y=int(round((h-side)*(yi+.5)/n))
            if (x,y) not in pos:pos.append((x,y))
            if len(pos)==count:return pos
    return pos

def flat_edge_maps(input_rgb):
    gray=cv2.cvtColor(input_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur=cv2.GaussianBlur(gray,(5,5),1)
    gx=cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
    grad=np.hypot(gx,gy)
    return gray,grad,grad<=np.percentile(grad,20),grad>=np.percentile(grad,90)

def auxiliary(input_rgb,output,aux):
    g,grad,flat,edge=aux
    og=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY).astype(np.float32)
    base_hpf=g-cv2.GaussianBlur(g,(5,5),1)
    out_hpf=og-cv2.GaussianBlur(og,(5,5),1)
    denom=float(np.sqrt(np.mean(base_hpf[flat]**2)))
    ratio=float(np.sqrt(np.mean(out_hpf[flat]**2))/denom) if denom>1e-6 else float('nan')
    smooth=cv2.GaussianBlur(og,(5,5),1)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    edge_ratio=float(np.hypot(gx,gy)[edge].mean()/max(float(grad[edge].mean()),1e-8))
    return ratio,edge_ratio,float(np.abs(input_rgb.astype(np.float32)-output.astype(np.float32)).mean())

def quality(reference,output):
    if reference is None:return None,None,None
    return (float(np.mean((reference.astype(np.float32)-output.astype(np.float32))**2)),
            float(peak_signal_noise_ratio(reference,output,data_range=255)),
            float(structural_similarity(reference,output,channel_axis=2,data_range=255)))

def assess_alignment(low,high):
    if low.shape!=high.shape:return 'size_mismatch',float('nan'),float('nan'),0.0
    h,w=low.shape[:2]
    s=min(1.0,512/max(h,w))
    def prep(img):
        g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        if s<1:g=cv2.resize(g,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g.astype(np.float32),(9,9),1.5)
    lo=prep(low);hi=prep(high)
    try:
        (dx,dy),response=cv2.phaseCorrelate(lo,hi)
        dx/=s;dy/=s
    except cv2.error:return 'failed_estimation',float('nan'),float('nan'),0.0
    if not all(math.isfinite(v) for v in (dx,dy,response)):return 'failed_estimation',dx,dy,response
    # A translation over 1.5 native pixels can make PSNR/SSIM misleading.
    ok=abs(dx)<=1.5 and abs(dy)<=1.5 and response>=.05
    return ('aligned_estimate' if ok else 'uncertain_alignment'),dx,dy,response

def write_comparison(noisy,low,variants,dest,title):
    tile=256;labels=['Real noisy input','Low ISO reference','Gaussian k3','Median k3','Bilateral d5']
    pictures=[noisy,low]+variants
    canvas=Image.new('RGB',(tile*len(pictures),tile+36),(245,247,250));dr=ImageDraw.Draw(canvas)
    for i,(img,label) in enumerate(zip(pictures,labels)):
        im=Image.fromarray(img);im=im.resize((tile,tile),Image.Resampling.NEAREST)
        canvas.paste(im,(i*tile,36));dr.text((i*tile+5,7),label,fill=(28,43,61))
    dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)

def inspect_files(root,out):
    allrows=[];missing=[]
    for item in all_files():
        path=root/item['family']/item['scene']/item['filename']
        exists=path.exists() and path.stat().st_size>500
        allrows.append((item['family'],item['scene'],item['iso'],item['filename'],exists,path.stat().st_size if exists else 0))
        if not exists:missing.append(str(path))
    with open(out/'数据完整性核验.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['family','scene','iso','filename','present','bytes']);w.writerows(allrows)
    return allrows,missing

def read_existing(path):
    if not path.exists():return set(),[]
    with open(path,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    return { (r['family'],r['scene'],r['iso'],r['patch'],r['method'],r['params']) for r in rows },rows

def stats(out,root,allrows,missing,requested,finished,new_rows,skipped,patches_per_image,debug=False):
    records_file=out/'逐图逐参数实测.csv'
    with open(records_file,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    buckets=collections.defaultdict(list)
    for r in rows:
        if r['method']=='Input':continue
        buckets[(r['method'],r['params'])].append(r)
    result=[]
    for (method,params),values in sorted(buckets.items()):
        entry={'method':method,'params':params,'patch_records':len(values)}
        for col in ['psnr_to_lowISO','ssim_to_lowISO','psnr_gain_over_input','flat_hf_ratio','edge_gradient_ratio','mean_abs_change','run_ms']:
            vals=[]
            for v in values:
                try:
                    x=float(v[col]);
                    if math.isfinite(x):vals.append(x)
                except (ValueError,TypeError):pass
            entry['n_'+col]=len(vals)
            entry['mean_'+col]=round(float(np.mean(vals)),5) if vals else ''
        result.append(entry)
    if result:
        with open(out/'分方法参数汇总.csv','w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(result[0].keys()));w.writeheader();w.writerows(result)
    # Generate evidence charts from the measured CSV. Blank / misaligned
    # reference scores are excluded rather than treated as zeros.
    if result:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for field,ylab,title in [
            ('mean_psnr_gain_over_input','PSNR gain (dB)','PSNR improvement vs noisy input'),
            ('mean_edge_gradient_ratio','Edge gradient ratio','Edge preservation proxy'),
            ('mean_run_ms','Runtime per patch (ms)','Filtering time')]:
            points=[(x['method']+' '+x['params'],float(x[field])) for x in result if x[field]!='']
            if not points:continue
            fig,ax=plt.subplots(figsize=(max(9,len(points)*.95),4.7))
            xx=np.arange(len(points))
            ax.bar(xx,[p[1] for p in points]);ax.set_xticks(xx,[p[0] for p in points],rotation=36,ha='right',fontsize=8)
            ax.set_ylabel(ylab);ax.set_title(title);ax.grid(axis='y',alpha=.2)
            fig.tight_layout();fig.savefig(out/(field+'.png'),dpi=145);plt.close(fig)
    # Additional group tables support analysis by actual ISO and scene.
    iso_groups=collections.defaultdict(list)
    scene_groups=collections.defaultdict(list)
    for r in rows:
        if r['method']=='Input':continue
        iso_groups[(r['iso'],r['method'],r['params'])].append(r)
        scene_groups[(r['family'],r['scene'],r['method'],r['params'])].append(r)
    for path,groups,names in [(out/'分ISO汇总.csv',iso_groups,['iso','method','params']),
                             (out/'分场景汇总.csv',scene_groups,['family','scene','method','params'])]:
        with open(path,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f)
            w.writerow(names+['patch_records','valid_psnr_n','mean_psnr','mean_psnr_gain','mean_ssim','mean_run_ms'])
            for keys,rr in sorted(groups.items()):
                def average(key):
                    vals=[]
                    for v in rr:
                        try:
                            n=float(v[key])
                            if math.isfinite(n):vals.append(n)
                        except (ValueError,TypeError):pass
                    return (round(float(np.mean(vals)),5) if vals else '',len(vals))
                ps,n=average('psnr_to_lowISO');gain,_=average('psnr_gain_over_input');ss,_=average('ssim_to_lowISO');tm,_=average('run_ms')
                w.writerow(list(keys)+[len(rr),n,ps,gain,ss,tm])
    expected_variant_count=sum(len(x['isos'])-1 for x in records())
    expected_records=expected_variant_count*patches_per_image*(len(PSET)+1)
    is_complete=(not missing and not skipped and not debug and len(rows)==expected_records and finished==expected_variant_count)
    status={'dataset':'NIND (Wikimedia Commons snapshot 2021-07-17)',
      'expected_original_images':len(allrows),'present_original_images':len(allrows)-len(missing),
      'missing_original_images':len(missing),'expected_scenes':len(list(records())),
      'requested_noisy_variants':requested,'finished_variant_pairs_this_run':finished,
      'new_filter_patch_records_this_run':new_rows,'total_csv_rows':len(rows),
      'all_original_images_available':not bool(missing),
      'all_experiments_complete':is_complete,'expected_noisy_variants':expected_variant_count,
      'expected_csv_rows_at_default_patches':expected_records,'skipped_count':len(skipped),
      'method':'Three classical filters with three parameter sets, plus Input baseline',
      'no_synthetic_noise':True,'reference':'lowest ISO capture, not absolute clean ground truth',
      'quality_protocol':'PSNR/SSIM only for matched-size pairs with <=1.5 px estimated shift and correlation response >=0.05',
      'methodology':'Full original file coverage, 4 native-resolution 256x256 deterministic patches per non-reference file by default',
      'bit_depth':'16-bit PNG converted to 8-bit by /257; results in 8-bit RGB domain, original data unchanged'}
    (out/'完成状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    return status

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',default='原始NIND');a.add_argument('--out',default='全量实验结果')
    a.add_argument('--allow-partial',action='store_true',help='only process present data; never label output full')
    a.add_argument('--patch-size',type=int,default=256);a.add_argument('--patches',type=int,default=4)
    a.add_argument('--max-variants',type=int,default=0,help='nonzero is debug only, cannot complete full set')
    a.add_argument('--preview-limit',type=int,default=16)
    args=a.parse_args();root=pathlib.Path(args.root);out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    allrows,missing=inspect_files(root,out)
    print('Original images',len(allrows)-len(missing),'/',len(allrows),flush=True)
    if missing and not args.allow_partial:
        (out/'下载缺失文件清单.txt').write_text('\n'.join(missing),encoding='utf-8')
        (out/'完成状态.json').write_text(json.dumps({'complete':False,'reason':'Original NIND files missing',
              'expected':len(allrows),'available':len(allrows)-len(missing)},ensure_ascii=False,indent=2),encoding='utf-8')
        print('Incomplete NIND dataset, stopped before computing. Download originals or use --allow-partial.',file=sys.stderr)
        return 3
    output_csv=out/'逐图逐参数实测.csv'
    done,_=read_existing(output_csv)
    if not output_csv.exists():
        with open(output_csv,'w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerow(FIELDS)
    total_req=0;complete=0;new_rows=0;preview=0
    skipped=[]
    for r in records():
        lowpath=root/r['family']/r['scene']/filename(r,r['reference_iso'])
        if not lowpath.exists():continue
        try:low,lowbits=read_rgb_u8(lowpath)
        except Exception as e:skipped.append((r['scene'],'reference',str(e)));continue
        for iso in r['isos'][1:]:
            highpath=root/r['family']/r['scene']/filename(r,iso)
            if not highpath.exists():continue
            total_req+=1
            if args.max_variants and total_req>args.max_variants:break
            try:high,bits=read_rgb_u8(highpath)
            except Exception as e:skipped.append((r['scene'],iso,str(e)));continue
            status,dx,dy,resp=assess_alignment(low,high)
            if status=='size_mismatch':skipped.append((r['scene'],iso,status));continue
            patches=normalized_patch_positions(high.shape[0],high.shape[1],args.patch_size,args.patches)
            if not patches:skipped.append((r['scene'],iso,'too_small'));continue
            complete+=1
            basekey=(r['family'],r['scene'],iso)
            for patch_i,(x,y) in enumerate(patches,1):
                ps=f'P{patch_i}';src=high[y:y+args.patch_size,x:x+args.patch_size].copy();ref=low[y:y+args.patch_size,x:x+args.patch_size].copy()
                reference=ref if status=='aligned_estimate' else None
                a=flat_edge_maps(src)
                imse,ipsnr,issim=quality(reference,src)
                for method,code,params in [('Input','no_filter',{})]+PSET:
                    key=(*basekey,ps,method,code)
                    if key in done:continue
                    start=time.perf_counter()
                    result=filter_rgb(src,method,params)
                    elapsed=(time.perf_counter()-start)*1000
                    mse,pn,ss=quality(reference,result)
                    fr,er,ac=auxiliary(src,result,a)
                    rec={
                      'family':r['family'],'scene':r['scene'],'reference_iso':r['reference_iso'],'iso':iso,
                      'input_path':str(highpath),'reference_path':str(lowpath),'patch':ps,'x':x,'y':y,'patch_size':args.patch_size,
                      'source_bits':bits,'pair_status':status,'shift_x_native_px':dx,'shift_y_native_px':dy,'phase_response':resp,
                      'method':method,'params':code,'mse_to_lowISO':mse if mse is not None else '',
                      'psnr_to_lowISO':pn if pn is not None else '','ssim_to_lowISO':ss if ss is not None else '',
                      'psnr_gain_over_input':pn-ipsnr if pn is not None else '',
                      'flat_hf_ratio':fr,'edge_gradient_ratio':er,'mean_abs_change':ac,'run_ms':elapsed,
                      'note':'8-bit converted from 16-bit' if bits==16 else ''}
                    with open(output_csv,'a',encoding='utf-8-sig',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=FIELDS);w.writerow(rec)
                    done.add(key);new_rows+=1
                if preview<args.preview_limit and patch_i==1:
                    variants=[filter_rgb(src,m,p) for m,c,p in [PSET[0],PSET[3],PSET[6]]]
                    dest=out/'可视化对比图'/f'{r["family"]}_{r["scene"]}_ISO{iso}.png'
                    write_comparison(src,ref,variants,dest,basekey);preview+=1
            print(f'[{complete}] {r["scene"]}, ISO {iso} : {len(patches)} native-size patches, alignment={status}',flush=True)
        if args.max_variants and total_req>args.max_variants:break
    if skipped:
        with open(out/'跳过文件原因.csv','w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['scene','iso','reason']);w.writerows(skipped)
    stat=stats(out,root,allrows,missing,total_req,complete,new_rows,skipped,args.patches,bool(args.max_variants))
    if args.max_variants:stat['all_experiments_complete']=False;stat['debug_limit']=args.max_variants
    (out/'完成状态.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stat,ensure_ascii=False,indent=2));return 0 if stat['all_experiments_complete'] else 2

if __name__=='__main__':sys.exit(main())
