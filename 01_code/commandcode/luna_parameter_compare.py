"""Paired Luna API parameter tuning vs frozen prompt optimization; no local fitting."""
import argparse,csv,html,json,msvcrt,os,time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import prompt_ablation as p

OLD=p.OUT; INTEGRATED=p.DATA/'luna_integrated_prompt200_20261116'
OUT=p.DATA/'luna_parameter_vs_prompt_20261116';p.OUT=OUT
ARMS=('P0_repeat','P0_medium','P0_high','P0_temp0','P2_repeat');p.ARMS=ARMS
NAMES={'P0_repeat':'原提示词·原参数（本轮重跑）','P0_medium':'原提示词·medium推理','P0_high':'原提示词·high推理','P0_temp0':'原提示词·temperature=0',
       'P2_repeat':'优化提示词P2·原参数（本轮重跑）','TFI':'本地TFI','P0':'历史原提示词P0','P1':'历史局部上下文P1','P2':'历史行为摘要P2','P3':'历史训练集示例P3','P4':'历史整合提示词P4'}
def sha(path):return p.api.sha(Path(path).read_bytes())
def save(name,obj):p.api.save(OUT/name,obj)
def prepare():
    if (OUT/'manifest.json').exists():
        for path,h in p.api.read(OUT/'manifest.json')['files'].items():assert sha(path)==h,path
        return
    for path,h in p.api.read(OLD/'manifest.json')['files'].items():assert sha(path)==h,path
    for option in ('medium','high','temp0'):assert p.api.read(OUT/'capability_standard_client'/f'{option}.json')['http_status']==200
    rows=p.api.read(OLD/'cohorts.json')['dev'];requests=p.api.read(OLD/'requests.json')['dev'];b0=requests['P0'];b2=requests['P2']
    assert len(rows)==len(b0)==len(b2)==200
    bodies={}
    for arm in ARMS:
        bodies[arm]=[]
        for i,r in enumerate(rows):
            b=json.loads(json.dumps(b2[i] if arm=='P2_repeat' else b0[i]))
            if arm=='P0_medium':b['reasoning_effort']='medium'
            if arm=='P0_high':b['reasoning_effort']='high'
            if arm=='P0_temp0':b['temperature']=0
            expected=b2[i] if arm=='P2_repeat' else b0[i]
            assert b['messages']==expected['messages']
            assert b['model']=='gpt-6-luna' and b['max_completion_tokens']==4096 and b['response_format']=={'type':'json_object'}
            changed=dict(r,truth=r['truth'].copy())
            for q in r['pos']:changed['truth'][q]=1234
            assert p.payload(changed,'P2' if arm=='P2_repeat' else 'P0')['messages']==expected['messages']
            bodies[arm].append(b)
    save('cohorts.json',{'dev':rows});save('requests.json',{'dev':bodies})
    spec={'utc':p.api.utc(),'model':'gpt-6-luna','provider':'CommandCode','flows':200,'mask_rate':.3,'hidden_slots':sum(len(r['pos']) for r in rows),
        'arms':{a:{k:bodies[a][0].get(k) for k in ('model','reasoning_effort','temperature','max_completion_tokens','response_format')} for a in ARMS},
        'prompt_choice':'P2 is frozen before this run, selected by prior prompt study Accuracy rule. P1/P3/P4 historical results also retained; no further prompt edits.',
        'concurrency':6,'budget':'1000 task response slots (five arms x 200), max3 transport attempts each; synthetic probes recorded separately. No retries for invalid HTTP200 outputs.',
        'retries':'Transport/429/5xx only. Auth/payment/permission/config errors stop dispatch. Requests interleaved by flow and arm.',
        'invalid':'Retain all200, fill only missing slots with zero; observed values unchanged; report JSON valid rate and failures.',
        'comparison':'Same Luna, same flows/masks/classifiers. Fresh P0 and P2 repeat to expose historical drift/stochasticity. Parameter arms preserve P0 messages byte-for-byte.',
        'parameter_selection':'Among medium/high/temp0 with mean Accuracy>P0_repeat and macroF1>=P0_repeat, rank by Accuracy then macroF1 then lower MAE; otherwise no improved parameter candidate. Report all arms, not only winner.',
        'scope':'API inference only; no model training or change to regression/numeric heads. No full1000-flow or new confirmation expansion in this batch.',
        'limitations':'Previously used historical test dev200; exploration only. One response per flow per arm. Multiple trials, unequal compute, provider-declared model identity. HTTP acceptance does not prove actual upstream parameter effect.',
        'engineering_note':'Initial synthetic probes with default Python User-Agent hit HTTP403 error1010. Reused established NIDS-Research-CommandCode/1.0 client header: all three HTTP200. Both probe sets retained.',
        'documentation':'https://commandcode.ai/docs/provider','selection_frozen_before_requests':True}
    save('protocol.json',spec)
    files={str(x):sha(x) for x in [Path(__file__),Path(p.__file__),Path(p.api.__file__),OLD/'cohorts.json',OLD/'requests.json',OUT/'cohorts.json',OUT/'requests.json',OUT/'protocol.json']}
    for a in ('P0','P1','P2','P3','TFI'):
        for suffix in ('_scores.json','_predictions.json'):
            path=OLD/'dev'/(a+suffix);files[str(path)]=sha(path)
    for suffix in ('_scores.json','_predictions.json'):
        path=INTEGRATED/'dev'/('P4'+suffix);files[str(path)]=sha(path)
    files.update(p.api.read(p.DATA/'joint_head_dev200_20261116/original_head_tuning/protocol.json')['source_hashes'])
    save('manifest.json',{'files':files})
    save('input_audit.json',{'passed':True,'same200_flows':True,'same_masks':True,'P0_messages_unchanged':True,'P2_messages_unchanged':True,'hidden_truth_mutation_pass':True,'no_local_training':True})

def tab(rs):
    return '<table>'+''.join('<tr>'+''.join(f'<{tag}>'+html.escape(f'{v:.2f}' if isinstance(v,float) else str(v))+f'</{tag}>' for v in row)+'</tr>' for tag,row in [('th',rs[0])]+[('td',r) for r in rs[1:]])+'</table>'

def report():
    rows=p.api.read(OUT/'cohorts.json')['dev'];reqs=p.api.read(OUT/'requests.json')['dev'];y=[r['label'] for r in rows]
    results={a:p.api.read(OUT/'dev'/(a+'_scores.json')) for a in ARMS};history={a:p.api.read(OLD/'dev'/(a+'_scores.json')) for a in ('P0','P1','P2','P3','TFI')}
    history['P4']=p.api.read(INTEGRATED/'dev/P4_scores.json');usage={};truth=np.array([r['truth'][q] for r in rows for q in r['pos']]);audit={}
    for a,s in results.items():
        saved=p.api.read(OUT/'dev'/(a+'_predictions.json'));values=saved['values'];counts=Counter();times=[];errors=Counter();models=Counter()
        for i,(r,vs) in enumerate(zip(rows,values)):
            z=p.api.read(OUT/'dev'/a/f'{i:04d}.json');assert z['request_sha256']==p.api.sha(p.api.enc(reqs[a][i]))
            vv,error=p.api.validate(z['response'],len(r['pos'])) if z['http_status']==200 else (None,'http_failure');assert vv==vs and error==z['validation_error']
            if z['http_status']==200:assert z['response'].get('model')=='gpt-6-luna'
            models[str(z['response'].get('model'))]+=1;errors[str(error)]+=1;times.append(sum(at['seconds'] for at in z['attempts']))
            for k,v in z['response'].get('usage',{}).items():
                if isinstance(v,(int,float)):counts[k]+=v
        pred=np.array([v for r,vs in zip(rows,values) for v in (vs if vs is not None else [0]*len(r['pos']))])
        assert abs(np.abs(np.abs(truth)-np.abs(pred)).mean()-s['magnitude_mae'])<1e-9
        for k,cs in saved['classes'].items():
            assert abs(100*accuracy_score(y,cs)-s['classifiers'][k]['accuracy'])<1e-9
            assert abs(100*f1_score(y,cs,labels=range(6),average='macro',zero_division=0)-s['classifiers'][k]['macro_f1'])<1e-9
        usage[a]={'reported_tokens':dict(counts),'mean_seconds':float(np.mean(times)),'median_seconds':float(np.median(times)),'validation_errors':dict(errors),'response_models':dict(models)}
        audit[a]={'passed':True,'flows':200,'requests_verified':200}
    for path,h in p.api.read(OUT/'manifest.json')['files'].items():assert sha(path)==h,path
    base=results['P0_repeat'];eligible=[a for a in ('P0_medium','P0_high','P0_temp0') if results[a]['mean_accuracy']>base['mean_accuracy'] and results[a]['mean_macro_f1']>=base['mean_macro_f1']]
    chosen=sorted(eligible,key=lambda a:(-results[a]['mean_accuracy'],-results[a]['mean_macro_f1'],results[a]['magnitude_mae'],a))[0] if eligible else None
    save('selection.json',{'eligible':eligible,'selected':chosen,'rule':p.api.read(OUT/'protocol.json')['parameter_selection'],'confirmation_run':False,'full_run':False})
    save('comparison.json',{'current':results,'historical':history,'usage':usage});save('result_audit.json',{'passed':True,'sources_unchanged':True,'arms':audit})
    cols=['方案','Accuracy %','macro-F1 %','幅值MAE/字节','带符号MAE','方向正确率 %','格式有效率 %','平均响应秒']
    table=[cols]+[[NAMES[a],s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],s['signed_mae'],s['direction_pct'],s['valid_pct'],usage[a]['mean_seconds']] for a,s in results.items()]
    tf=history['TFI'];table.append([NAMES['TFI'],tf['mean_accuracy'],tf['mean_macro_f1'],tf['magnitude_mae'],tf['signed_mae'],tf['direction_pct'],tf['valid_pct'],'本地'])
    hist=[cols[:-1]]+[[NAMES[a],s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],s['signed_mae'],s['direction_pct'],s['valid_pct']] for a,s in history.items()]
    details=[['方案','分类器','Accuracy %','macro-F1 %','Miuref召回 %','SMB召回 %']]
    for a,s in {**results,'TFI':history['TFI']}.items():
        for c,z in s['classifiers'].items():details.append([NAMES[a],c,z['accuracy'],z['macro_f1'],z['recall_by_class'][4],z['recall_by_class'][1]])
    for name,rs in [('summary.csv',table),('historical_summary.csv',hist),('classifiers.csv',details)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rs)
    body='<html lang="zh"><meta charset="utf-8"><style>body{font:16px Arial,"Microsoft YaHei";margin:32px;line-height:1.7}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;margin:24px 0}th{border-bottom:1px solid}td,th{padding:9px}</style><h1>Luna：调用参数调优与提示词优化</h1><p>同一200条D4开发流、原30%缺包位置、2045个缺包，六个冻结分类器等权平均。仅调用API，未训练本地模型。缺少或格式无效回答按缺包位填零，所有流保留。</p>'+tab(table)+'<h2>历史提示词结果（单独列示）</h2>'+tab(hist)+'<h2>六分类器明细</h2>'+tab(details)
    body+='<p>本轮参数候选：'+str(chosen)+'。P2在本轮前固定为原提示词实验按准确率选择的版本，P1历史macro-F1更高；不把二者混为同一提示词。原参数P0和优化P2均重新调用，历史分数变化属于调用复现差异。</p><p>接口HTTP200仅证明请求被接受；平台未返回可独立验证的上游参数执行证明。模型身份为平台返回的gpt-6-luna。medium/high不是已经证实的实际额外推理量。历史开发池反复使用，本轮是探索比较，不是独立最终测试或超过原文的证据。</p><p><a href="protocol.json">预登记</a> · <a href="result_audit.json">核验</a> · <a href="comparison.json">指标与token用量</a></p></html>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    (OUT/'CHANGELOG.txt').write_text(json.dumps(p.api.read(OUT/'protocol.json'),ensure_ascii=False,indent=2)+'\n'+json.dumps(p.api.read(OUT/'comparison.json'),ensure_ascii=False,indent=2),encoding='utf-8')
    save('complete.json',{'utc':p.api.utc(),'selected_parameter_arm':chosen,'slots':1000,'no_training':True})
    print('COMPLETE',json.dumps({a:{k:s[k] for k in ('mean_accuracy','mean_macro_f1','magnitude_mae','valid_pct')} for a,s in results.items()}),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    prepare()
    if (OUT/'complete.json').exists():print('Already complete');return
    save('started.json',{'utc':p.api.utc(),'pid':os.getpid()});print('START five arms x 200, concurrency6, fixed prompts',flush=True)
    p.run_api('dev',ARMS);report();save('progress.json',{'utc':p.api.utc(),'state':'complete','counts':{a:200 for a in ARMS}})
if __name__=='__main__':main()
