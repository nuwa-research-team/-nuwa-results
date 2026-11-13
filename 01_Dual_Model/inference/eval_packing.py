"""Native chat-template offset checked soft-prefix assembly, PDF Appendix B.

Single sample by design (micro_batch=1). Does not implement training or decide
splits. Labels/answer tokens never enter the traffic encoders. No new vocab.
"""
import json
import torch
from torch.nn import functional as F

BEHAVIOR_MARKER = '[BEHAVIOR_VECTOR_SLOT]\n'
PACKET_MARKER = '[PACKET_VECTOR_SLOT]\n'


def template_parts(tokenizer, n, queries, canonical=True):
    assert queries == sorted(set(queries)) and all(0 <= x < n for x in queries)
    direction = 'canonical packet direction' if canonical else 'packet direction relative to the flow origin'
    user = ('Known valid packet slots: '+str(n)+'. Behavior representations:\n' + BEHAVIOR_MARKER +
            'Packet representations:\n' + PACKET_MARKER +
            'Missing slot indices, in ascending order: '+json.dumps(queries, separators=(',', ':'))+
            '. Return only one compact JSON array of signed IP packet lengths, in this query order. '
            'Each item must be an integer with absolute value in [40,65535]. No explanation.')
    messages = [dict(role='system', content='Restore missing signed IP packet lengths from visible '
                     f'traffic and known slots. Sign denotes {direction}. '
                     'Use all given context. Output only the requested integer JSON array.'),
                dict(role='user', content=user)]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    # Explicitly prove fast-tokenizer text encoding equals actual chat-template ids.
    native = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    assert ids == native, 'Chat-template tokenization mismatch'
    spans = []
    for marker in (BEHAVIOR_MARKER, PACKET_MARKER):
        assert text.count(marker) == 1
        start = text.index(marker)
        end = start+len(marker)
        chosen = [i for i, (a, b) in enumerate(offsets) if b>start and a<end]
        assert chosen and offsets[chosen[0]][0] == start and offsets[chosen[-1]][1] == end, \
            ('Soft-token insertion cuts a tokenizer token', marker,
             [offsets[i] for i in chosen], (start, end))
        assert all(start<=offsets[i][0] and offsets[i][1]<=end for i in chosen)
        spans.append((chosen[0], chosen[-1]+1))
    (a0, a1), (b0, b1) = spans
    assert a1 <= b0
    segments = [ids[:a0], ids[a1:b0], ids[b1:]]
    # Preserve exact ids outside the two placeholder spans; no adjacent text is dropped.
    assert segments[0]+ids[a0:a1]+segments[1]+ids[b0:b1]+segments[2] == ids
    return dict(segments=segments, full_template=text, full_template_ids=ids, marker_token_spans=spans)


def pack_single(tokenizer, embedding, packet_embeds, behavior_embeds, n, queries,
                answer=None, max_total_tokens=8192, canonical=True):
    """packet_embeds=N,D; behavior_embeds=K,D; only real (unpadded) soft tokens."""
    assert packet_embeds.ndim == behavior_embeds.ndim == 2 and packet_embeds.shape[0] == n
    assert packet_embeds.shape[1] == behavior_embeds.shape[1] == embedding.weight.shape[1]
    layout = template_parts(tokenizer, n, queries, canonical)
    device, dtype = embedding.weight.device, embedding.weight.dtype
    texts = [embedding(torch.tensor(ids, device=device, dtype=torch.long)) for ids in layout['segments']]
    prefix = torch.cat([texts[0], behavior_embeds.to(dtype), texts[1], packet_embeds.to(dtype), texts[2]], dim=0)
    prefix_length = prefix.shape[0]
    labels = torch.full((prefix_length,), -100, device=device, dtype=torch.long)
    answer_ids = []
    if answer is not None:
        assert len(answer) == len(queries) and all(type(v) is int for v in answer)
        assert all(40 <= abs(v) <= 65535 for v in answer)
        text = json.dumps(answer, separators=(',', ':'))
        answer_ids = tokenizer(text, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        target = torch.tensor(answer_ids, device=device, dtype=torch.long)
        prefix = torch.cat([prefix, embedding(target)], dim=0)
        labels = torch.cat([labels, target])
    if len(prefix) > max_total_tokens:
        raise ValueError('Context budget exceeded; no sample truncation allowed')
    length = len(prefix)
    return dict(inputs_embeds=prefix.unsqueeze(0), labels=labels.unsqueeze(0),
                attention_mask=torch.ones(1, length, device=device, dtype=torch.long),
                position_ids=torch.arange(length, device=device).unsqueeze(0),
                prefix_length=prefix_length, answer_token_count=len(answer_ids), layout=layout)


def causal_answer_loss(logits, labels):
    """Reference per-sample shifted CE, full-logits path for numerical validation."""
    gold = labels[:, 1:]
    token = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), gold,
                            ignore_index=-100, reduction='none')
    count = gold.ne(-100).sum(1)
    assert bool((count > 0).all())
    return (token.sum(1)/count).mean()


def answer_loss_from_hidden(hidden, lm_head, labels):
    """Same loss with answer-only head projection, avoiding full-prefix logits."""
    gold = labels[:, 1:]
    keep = gold.ne(-100)
    assert bool((keep.sum(1)>0).all())
    selected = hidden[:, :-1][keep]
    scores = lm_head(selected).float()
    token_loss = F.cross_entropy(scores, gold[keep], reduction='none')
    batch_index = torch.arange(len(labels), device=labels.device)[:, None].expand_as(keep)[keep]
    sums = torch.zeros(len(labels), device=token_loss.device).scatter_add(0, batch_index, token_loss)
    return (sums/keep.sum(1)).mean()


def parse_p1(raw_text, query_count):
    """No repairs, no retry, no nearest-value snapping; invalids stay invalid."""
    try:
        values = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return dict(status='invalid_json', parsed_values=None)
    if not isinstance(values, list) or len(values) != query_count:
        return dict(status='wrong_count_or_type', parsed_values=values)
    if not all(type(v) is int for v in values):
        return dict(status='non_integer', parsed_values=values)
    if not all(40 <= abs(v) <= 65535 for v in values):
        return dict(status='illegal_length', parsed_values=values)
    return dict(status='valid', parsed_values=values)
