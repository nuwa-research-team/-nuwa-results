"""P4: one frozen integrated instruction, same P2 visible-only inputs, 200 dev flows."""
import argparse,csv,html,json,msvcrt,os
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import prompt_ablation as p

OLD=p.OUT
OUT=p.DATA/'luna_integrated_prompt200_20261116'
PROMPT='''You reconstruct missing signed IP packet lengths in ONE bidirectional network flow.

TASK AND CONVENTIONS
The signed array is in original chronological packet order. A positive or negative sign identifies one of the two directions; it does not identify an application, client, or server. Absolute magnitude is IP packet length in bytes, clipped at 1600. Every null represents exactly one missing packet. Recover only those missing signed lengths. Observed values and their directions are fixed.

HOW TO READ THE INPUT
missing_positions_zero_based specifies the exact output order. missing_count is the exact required number of output integers. Every index, neighbor distance, gap_start and gap_size refers to original packet slots, including missing ones. A neighbor triplet is [original_index, observed_signed_length, distance_in_original_slots]. Several missing slots may share the same boundary observations; each still needs its own answer.
Behavior summaries are computed ONLY from observed packets. They are redundant views of signed, not additional packets or hidden answers. Frequency entries are [value, count]. A run entry is [start_index, end_index, direction, observed_packet_count]. A run stops at a missing slot or an observed direction change. Window support records how many packets were observed, not the complete window's unknown distribution. If a summary is ambiguous, consult the signed array.

LOCAL EVIDENCE FIRST
For each missing slot, use nearby observed packets on BOTH sides and their actual slot distances. Compare the neighboring directions, repeated lengths and surrounding send/receive pattern. Close observations are usually more relevant than a whole-flow majority; distant observations and sparse windows provide weaker evidence. Missing slots between observations do not prove a direction switch or continuous same-direction run. At a boundary with only one observed side, use the available context without inventing the absent side.

DIRECTION AND MAGNITUDE
Determine a plausible direction from the local sequence before selecting a magnitude. Preserve the supplied direction convention throughout the flow. When signs alternate near a gap, consider where the missing packet fits in that alternation; when same-direction packets recur, consider continuation, but do not force either pattern from one neighbor pair alone.
For the chosen direction, compare plausible short and long packet lengths using local repetition, same-direction observed lengths, run boundaries and nearby windows. Direction-specific length frequencies can support a candidate. Global frequency is only a fallback when local evidence is weak. Repeated long packets may support another long packet; short packets may belong to a different part of the exchange. Neither is a universal rule. Do not always select the most frequent length, average two incompatible short/long modes, or assume every packet inside one missing run is identical. Observed lengths are useful candidates, but a genuinely different length remains allowed when the context supports it.

JOINT CONSISTENCY
Consider adjacent missing slots together so that their proposed directions and lengths form a plausible local sequence. Keep each slot's separate position; do not collapse a multi-packet gap into one value. Use summaries to support the local reconstruction, not to override strong local evidence. Do not infer or output application labels. Do not output a rationale.

EXACT OUTPUT CONTRACT
Return exactly one JSON object with exactly one key, "values". Its value must be a flat array of exactly missing_count nonzero integers. Each integer must be in [-1600,1600]. Element j fills missing_positions_zero_based[j]. Include no observed packets, indices, counts, confidence scores, prose, markdown, or extra keys.
Before returning, check that the array length equals missing_count and the number of listed missing positions, that each listed position has exactly one value in order, and that every value satisfies the signed-integer range. Correct formatting or count mistakes before emitting the single JSON object.'''

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        for path,h in p.api.read(OUT/'manifest.json')['files'].items():assert p.api.sha(Path(path).read_bytes())==h,path
        return
    for path,h in p.api.read(OLD/'manifest.json')['files'].items():assert p.api.sha(Path(path).read_bytes())==h,path
    rows=p.api.read(OLD/'cohorts.json')['dev']
    old=p.api.read(OLD/'requests.json')['dev']['P2']
    requests=[]
    for r,b in zip(rows,old):
        req=json.loads(json.dumps(b));req['messages'][0]['content']=PROMPT
        assert req['messages'][1]==b['messages'][1]
        changed=dict(r,truth=r['truth'].copy())
        for pos in r['pos']:changed['truth'][pos]=1234
        assert p.payload(changed,'P2')['messages'][1]==req['messages'][1]
        requests.append(req)
    p.api.save(OUT/'cohorts.json',{'dev':rows});p.api.save(OUT/'requests.json',{'dev':{'P4':requests}})
    (OUT/'integrated_prompt_en.txt').write_text(PROMPT+'\n',encoding='utf-8')
    protocol={'utc':p.api.utc(),'arm':'P4','model':p.MODEL,'draws':200,'mask_rate':.3,
        'change':'Replace system instruction with integrated hierarchy: local evidence first; direction then magnitude; support-aware behavior; joint gap consistency; exact output count.',
        'input_identity':'P2 user payload byte-for-byte unchanged. Same local context, behavior summaries, flows and masks. No training demonstrations.',
        'decode':'Same 4096 max completion tokens, low reasoning, JSON-object mode, strict validation; six concurrent calls.',
        'budget':'200 new response slots, max 3 transport attempts each; no retries for invalid answers; no retraining.',
        'evaluation':'All 200 flows retained. Invalid outputs zero-filled at missing slots. Same six frozen classifiers and macro-F1 labels.',
        'baseline':'Reuse P0/P1/P2/P3/TFI development scores and saved predictions, no historical file changes.',
        'boundary':'Adaptive follow-up on the same development cohort after earlier prompt results. No new held-out confirmation in this bounded run.',
        'prompts_are_hypotheses':'Combining instructions does not guarantee combining their observed gains.'}
    p.api.save(OUT/'protocol.json',protocol)
    files=dict(p.api.read(OLD/'manifest.json')['files'])
    for f in [Path(__file__),OUT/'cohorts.json',OUT/'requests.json',OUT/'integrated_prompt_en.txt',OUT/'protocol.json']:
        files[str(f)]=p.api.sha(f.read_bytes())
    for a in ('P0','P1','P2','P3','TFI'):
        for suffix in ('_scores.json','_predictions.json'):
            f=OLD/'dev'/(a+suffix);files[str(f)]=p.api.sha(f.read_bytes())
    p.api.save(OUT/'manifest.json',{'files':files})
    p.api.save(OUT/'input_audit.json',{'passed':True,'requests':200,'same_P2_payload':True,'hidden_value_mutation_invariant':True,'hidden_slots':sum(len(r['pos']) for r in rows)})

def report_and_audit():
    rows=p.api.read(OUT/'cohorts.json')['dev'];y=np.array([r['label'] for r in rows])
    saved=p.api.read(OUT/'dev/P4_predictions.json');score=p.api.read(OUT/'dev/P4_scores.json')
    req=p.api.read(OUT/'requests.json')['dev']['P4'];truth=[];pred=[];usage=Counter();errors=Counter()
    for i,(r,v) in enumerate(zip(rows,saved['values'])):
        rec=p.api.read(OUT/'dev/P4'/f'{i:04d}.json')
        assert rec['request_sha256']==p.api.sha(p.api.enc(req[i]))
        if rec['http_status']==200:
            assert rec['response'].get('model')==p.MODEL
            values,error=p.api.validate(rec['response'],len(r['pos']))
            assert values==v and error==rec['validation_error']
        else:assert v is None
        truth.extend(r['truth'][j] for j in r['pos']);pred.extend(v if v is not None else [0]*len(r['pos']))
        errors[str(rec['validation_error'])]+=1
        for k,n in rec['response'].get('usage',{}).items():
            if isinstance(n,(int,float)):usage[k]+=n
    truth=np.array(truth);pred=np.array(pred)
    assert abs(np.mean(np.abs(np.abs(truth)-np.abs(pred)))-score['magnitude_mae'])<1e-9
    for k,c in saved['classes'].items():
        assert abs(accuracy_score(y,c)*100-score['classifiers'][k]['accuracy'])<1e-9
        assert abs(f1_score(y,c,labels=range(6),average='macro',zero_division=0)*100-score['classifiers'][k]['macro_f1'])<1e-9
    for path,h in p.api.read(OUT/'manifest.json')['files'].items():assert p.api.sha(Path(path).read_bytes())==h,path
    p.api.save(OUT/'result_audit.json',{'passed':True,'rows':200,'errors':dict(errors),'usage':dict(usage)})
    results={a:p.api.read(OLD/'dev'/(a+'_scores.json')) for a in ('P0','P1','P2','P3','TFI')};results['P4']=score
    delta={a:{k:score[k]-s[k] for k in ('mean_accuracy','mean_macro_f1','magnitude_mae')} for a,s in results.items() if a!='P4'}
    p.api.save(OUT/'comparison.json',{'results':results,'P4_minus':delta})
    cols=['Method','Accuracy (%)','Macro-F1 (%)','Magnitude MAE (bytes)','Valid (%)']
    table=[];details=[['Classifier','Method','Accuracy (%)','Macro-F1 (%)']]
    names={'P0':'原提示词','P1':'加局部上下文','P2':'再加行为摘要','P3':'再加训练集示例','P4':'完整整合提示词','TFI':'本地 TFI'}
    for a in ('P0','P1','P2','P3','P4','TFI'):
        s=results[a];table.append([a+' '+names[a],s['mean_accuracy'],s['mean_macro_f1'],s['magnitude_mae'],s['valid_pct']])
        for k,c in s['classifiers'].items():details.append([k,a,c['accuracy'],c['macro_f1']])
    for name,rs in [('summary.csv',[cols,*table]),('classifiers.csv',details)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rs)
    def tab(rs):
        return '<table>'+''.join('<tr>'+''.join('<'+('th' if j==0 else 'td')+'>'+html.escape(f'{v:.2f}' if isinstance(v,float) else str(v))+'</'+('th' if j==0 else 'td')+'>' for v in row)+'</tr>' for j,row in enumerate(rs))+'</table>'
    text='<html lang="zh"><meta charset="utf-8"><title>Luna 完整提示词测试</title><style>body{font:16px Arial,"Microsoft YaHei";margin:40px;line-height:1.7}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;margin:25px 0}th{border-bottom:1px solid}td,th{padding:10px}</style><h1>Luna 完整整合提示词：P4</h1><p>原开发组 200 条 D4 流，30% 原缺包位置，六个冻结分类器。Accuracy 与 macro-F1 为六分类器等权均值；包长 MAE 单位字节。P4 保留 P2 的输入，替换系统提示词以明确局部证据优先、先方向后幅值、行为统计支持数、连续缺口一致性和输出数量检查；不含训练示例。不重训，不加时间戳，不修改 TFI。旧版预测复用。</p>'+tab([cols,*table])+'<details><summary>六分类器明细</summary>'+tab(details)+'</details><p>所有样本均计分，无效响应在缺失位置填零。此轮为看过旧结果后的开发组后续试验，未进行新的独立确认；不能据此声称超过原论文。</p><p><a href="integrated_prompt_en.txt">完整英文提示词</a> · <a href="CHANGELOG.txt">改动记录</a></p></html>'
    (OUT/'comparison.html').write_text(text,encoding='utf-8')
    lines=['P4 完整提示词整合实验',json.dumps(p.api.read(OUT/'protocol.json'),ensure_ascii=False,indent=2),
           '本轮指标：'+json.dumps(score,ensure_ascii=False,indent=2),'相对旧版变化：'+json.dumps(delta,ensure_ascii=False,indent=2),
           '调用 token（非计费金额）：'+json.dumps(dict(usage)),'错误统计：'+json.dumps(dict(errors))]
    (OUT/'CHANGELOG.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'score':score,'delta':delta,'usage':dict(usage)},ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    prepare()
    if args.prepare:return
    p.OUT=OUT
    p.api.save(OUT/'started.json',{'utc':p.api.utc(),'pid':os.getpid()})
    p.run_api('dev',('P4',))
    report_and_audit()
    p.api.save(OUT/'complete.json',{'utc':p.api.utc(),'draws':200,'audit_passed':True})
    p.api.save(OUT/'progress.json',{'state':'complete','utc':p.api.utc(),'draws':200})

if __name__=='__main__':main()
