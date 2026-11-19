"""下载 Wikimedia 官方 NIND 全部版本，断点续传与 SHA1 校验。

要求联网、至少约 40 GB 可用空间。仅下载公开图像，不运行网络代码。
    python scripts/download_nind.py --root 原始NIND
演示少量文件：--limit 2
注意：--thumb-width 1024 为缩略图、会改变噪声，不能称原尺寸实验。
"""
import argparse,csv,hashlib,json,os,pathlib,sys,time
import requests
from nind_manifest import all_files
API='https://commons.wikimedia.org/w/api.php'
HEAD={'User-Agent':'NINDTraditionalFilteringAcademic/1.0 (course image denoising; refer to Commons original metadata)'}

def write_rows(path,rows):
    fields=['family','scene','iso','filename','local_path','status','bytes','source_url','license_url','error']
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def sha1path(p):
    h=hashlib.sha1()
    with open(p,'rb') as fp:
        for block in iter(lambda:fp.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def getinfo(s,filename,width):
    q={'action':'query','format':'json','prop':'imageinfo','iiprop':'url|sha1|size|extmetadata',
       'titles':'File:'+filename.replace('_',' ')}
    if width:q.update(iiurlwidth=width)
    r=s.get(API,params=q,timeout=20);r.raise_for_status()
    data=next(iter(r.json()['query']['pages'].values()))
    if not data.get('imageinfo'):raise FileNotFoundError(filename)
    return data['imageinfo'][0]

def download(s,url,target,sha1=None):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size>100:
        if not sha1 or sha1path(target)==sha1:return 'cached',target.stat().st_size
        target.unlink()
    part=target.with_suffix(target.suffix+'.part')
    offset=part.stat().st_size if part.exists() else 0
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with s.get(url,stream=True,headers=headers,timeout=(20,120)) as r:
        r.raise_for_status()
        app=offset>0 and r.status_code==206
        with open(part,'ab' if app else 'wb') as f:
            for b in r.iter_content(1024*1024):
                if b:f.write(b)
    if part.stat().st_size<100:raise IOError('empty file')
    if sha1 and sha1path(part)!=sha1:
        part.unlink(missing_ok=True);raise IOError('SHA1 does not match Commons original')
    os.replace(part,target)
    return 'downloaded',target.stat().st_size

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='原始NIND')
    p.add_argument('--limit',type=int,default=0,help='Only for connectivity testing; 0=all official files')
    p.add_argument('--thumb-width',type=int,default=0,help='0=original files; >0=preview only, not true full-resolution experiment')
    p.add_argument('--max-errors',type=int,default=10)
    args=p.parse_args()
    root=pathlib.Path(args.root);root.mkdir(parents=True,exist_ok=True)
    files=list(all_files())
    if args.limit:files=files[:args.limit]
    (root/'下载计划.json').write_text(json.dumps({'planned':len(files),'source':'https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset',
      'is_complete_manifest':not bool(args.limit),'is_original_resolution':not bool(args.thumb_width)},ensure_ascii=False,indent=2),encoding='utf-8')
    sess=requests.Session();sess.headers.update(HEAD)
    rows=[];errors=0
    try:
        r=sess.get(API,params={'action':'query','format':'json','meta':'siteinfo'},timeout=12)
        r.raise_for_status()
    except requests.RequestException as e:
        (root/'下载失败原因.txt').write_text('无法连接 Wikimedia Commons，请在可联网的电脑上运行。\n'+str(e),encoding='utf-8')
        print('DOWNLOAD BLOCKED: Cannot connect to Wikimedia Commons:',e,file=sys.stderr)
        return 2
    for i,item in enumerate(files,1):
        filename=item['filename'];path=root/item['family']/item['scene']/filename
        row={k:item[k] for k in ['family','scene','iso','filename']}
        row.update(local_path=str(path),status='',bytes=0,source_url='',license_url=f'https://commons.wikimedia.org/wiki/File:{filename}',error='')
        try:
            info=getinfo(sess,filename,args.thumb_width)
            url=info.get('thumburl') if args.thumb_width else info['url']
            url=url or info['url']
            checksum=None if args.thumb_width else info.get('sha1')
            status,size=download(sess,url,path,checksum)
            row.update(status=status,bytes=size,source_url=url)
        except (requests.RequestException,IOError,KeyError,ValueError,FileNotFoundError) as e:
            row.update(status='error',error=repr(e));errors+=1
        rows.append(row)
        write_rows(root/'下载核验.csv',rows)
        print(f'[{i}/{len(files)}] {filename}: {row["status"]}',flush=True)
        if errors>=args.max_errors:
            print('Too many download errors; stopped safely, run again to resume.',file=sys.stderr);break
        time.sleep(.05)
    full={'requested':len(files),'verified':sum(r['status'] in ('downloaded','cached') for r in rows),
          'failed':sum(r['status']=='error' for r in rows),'pending':len(files)-len(rows),
          'complete':len(rows)==len(files) and errors==0,'original_files':args.thumb_width==0}
    (root/'下载完成情况.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(full,ensure_ascii=False))
    return 0 if full['complete'] else 1

if __name__=='__main__':sys.exit(main())
