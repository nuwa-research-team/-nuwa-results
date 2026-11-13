# -*- coding: utf-8 -*-
"""Parse original PCAPs into per-flow packet records (plan section 4).

Fields: frame.number, frame.time_epoch, tcp.stream, ip.src/dst, tcp.srcport/dstport,
ip.len, ip.flags.mf, ip.frag_offset. Keeps retransmissions, never clips ip.len,
excludes IPv6 / fragments / malformed lengths, and defines direction by the
canonical ordering of the two endpoints (A->B = +1).
"""
import argparse
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

FIELDS = ["frame.number", "frame.time_epoch", "tcp.stream", "ip.src", "ip.dst",
          "tcp.srcport", "tcp.dstport", "ip.len", "ip.flags.mf", "ip.frag_offset"]


def tshark_rows(pcap: Path):
    cmd = ["tshark", "-r", str(pcap), "-Y", "ip && tcp", "-T", "fields",
           "-E", "separator=\t"]
    for f in FIELDS:
        cmd += ["-e", f]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"tshark failed on {pcap}: {p.stderr[:300]}")
    for line in p.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < len(FIELDS):
            parts += [""] * (len(FIELDS) - len(parts))
        yield parts


def parse_capture(pcap: Path, capture_id: str, group_id: str):
    flows = {}
    stats = {"rows": 0, "dropped_fragment": 0, "dropped_badlen": 0,
             "dropped_no_stream": 0, "kept": 0}
    for parts in tshark_rows(pcap):
        stats["rows"] += 1
        fno, epoch, stream, src, dst, sp, dp, ln, mf, off = parts[:10]
        if not stream or not ln:
            stats["dropped_no_stream"] += 1
            continue
        if mf not in ("", "0") or (off and off not in ("0",)):
            stats["dropped_fragment"] += 1
            continue
        try:
            length = int(ln)
        except ValueError:
            stats["dropped_badlen"] += 1
            continue
        if not (40 <= length <= 65535):
            stats["dropped_badlen"] += 1
            continue
        try:
            ts = Decimal(epoch)
        except Exception:  # noqa: BLE001
            stats["dropped_badlen"] += 1
            continue
        a = (src, int(sp) if sp else 0)
        b = (dst, int(dp) if dp else 0)
        direction = 1 if a < b else -1
        rec = flows.setdefault(stream, {"endpoints": sorted([a, b]), "packets": []})
        rec["packets"].append((ts, int(fno), direction, length))
        stats["kept"] += 1

    out = []
    for stream, rec in flows.items():
        rec["packets"].sort(key=lambda x: (x[0], x[1]))
        t0 = rec["packets"][0][0]
        slots = []
        for slot, (ts, fno, d, ln) in enumerate(rec["packets"]):
            rel_us = int(((ts - t0) * Decimal(1000000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            slots.append({"slot": slot, "frame": fno, "signed_len": d * ln, "time_us": rel_us})
        out.append({"flow_id": f"{capture_id}:{stream}", "capture_id": capture_id,
                    "group_id": group_id, "n_packets": len(slots), "packets": slots})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with class sub-dirs of pcap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-rule", default="capture", choices=["capture", "class_date"])
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_flows, manifests = [], []
    for pcap in sorted(root.rglob("*.pcap")):
        cls = pcap.parent.name if pcap.parent != root else "unknown"
        capture_id = f"{cls}/{pcap.name}"
        group_id = pcap.stem if args.group_rule == "capture" else f"{cls}_{pcap.stem}"
        flows, stats = parse_capture(pcap, capture_id, group_id)
        for f in flows:
            f["label"] = cls
        all_flows.extend(flows)
        manifests.append({"capture": capture_id, "label": cls, "group_id": group_id,
                          **stats, "flows": len(flows),
                          "packets": sum(f["n_packets"] for f in flows)})
        print(json.dumps(manifests[-1]), flush=True)

    with open(out_path, "w") as fh:
        for f in all_flows:
            fh.write(json.dumps(f) + "\n")
    json.dump({"flows": len(all_flows), "captures": manifests,
               "packets": sum(m["packets"] for m in manifests)},
              open(str(out_path) + ".manifest.json", "w"), indent=2)
    print("total flows", len(all_flows), "packets", sum(m["packets"] for m in manifests))


if __name__ == "__main__":
    main()
