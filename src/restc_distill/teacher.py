"""Vertex AI hard-target teacher. One generation per input, strict JSON schema."""
from pathlib import Path
from .common import checked_rows, read, save, digest, parse_answer, write_lines


def extract_answer(response, positions):
    candidates = response.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ValueError("teacher did not return one complete STOP candidate")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought", False))
    return parse_answer(text, positions)


def generate(prepared, out, limit=None, split="train", predictions=None):
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    rows, manifest = checked_rows(prepared, split)
    if limit is not None:
        if not 0 < limit <= len(rows):
            raise ValueError("invalid teacher call limit")
        rows = rows[:limit]
    if predictions is not None and Path(predictions).exists():
        raise FileExistsError("prediction export already exists")
    teacher = manifest["config"]["teacher"]
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds.with_quota_project(teacher["project"]))
    url = (f'https://aiplatform.googleapis.com/v1/projects/{teacher["project"]}/locations/'
           f'{teacher["location"]}/publishers/google/models/{teacher["model"]}:generateContent')
    for index, row in enumerate(rows):
        path = Path(out) / (row["id"] + ".json")
        if path.exists():
            old = read(path)
            if old["request_sha256"] != row["request_sha256"] or old["teacher"] != teacher:
                raise ValueError("cached teacher identity mismatch")
            if old["error"] is not None:
                raise ValueError(f"cached failure: {path}; inspect it before starting a separate run")
            extract_answer(old["response"], row["positions"])
            continue
        if digest(row["teacher_request"]) != row["request_sha256"]:
            raise ValueError("request hash mismatch")
        # No best-of-N, answer repair, or substitution of model/schema.
        response = session.post(url, json=row["teacher_request"], timeout=(15, 180))
        try:
            raw = response.json()
        except ValueError:
            raw = {"non_json_response": True}
        error, answer = None, None
        try:
            if response.status_code != 200:
                raise ValueError("HTTP " + str(response.status_code))
            answer = extract_answer(raw, row["positions"])
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        save(path, {"id": row["id"], "teacher": teacher, "request_sha256": row["request_sha256"],
                    "http_status": response.status_code, "response": raw, "answer": answer, "error": error})
        print(f"teacher {index + 1}/{len(rows)}: {row['id'][:12]} {error or 'valid'}", flush=True)
        if error is not None:
            raise RuntimeError("teacher generation failed; raw response preserved")

    if predictions is not None:
        exported = []
        for row in rows:
            record = read(Path(out) / (row["id"] + ".json"))
            exported.append({"id": row["id"], "input_sha256": digest(row["messages"]),
                             "answer": extract_answer(record["response"], row["positions"]),
                             "teacher": teacher, "request_sha256": row["request_sha256"]})
        write_lines(predictions, exported)
