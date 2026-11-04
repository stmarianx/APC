"""Train and query APC's course-evidence encoder using lesson-isolated splits."""
import hashlib
import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "curriculum/generated/course_curriculum.jsonl"
WEIGHTS = ROOT / "checkpoints/course-encoder-v1.pt"


def features(text):
    vector = torch.zeros(1024)
    words = re.findall(r"[a-z0-9]+", text.lower())
    for word in words + [" ".join(pair) for pair in zip(words, words[1:])]:
        index = int.from_bytes(hashlib.blake2s(word.encode(), digest_size=4).digest(), "big")
        vector[index % len(vector)] += 1
    return F.normalize(vector, dim=0)


def passage(row):
    target = row["target"]
    if "evidence_text" in target:
        return target["evidence_text"]
    return " ".join([row["task"]["instruction"], target["formula"], *target["assumptions"]])


class CourseEncoder(nn.Module):
    def __init__(self, concepts):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1024, 64), nn.ReLU())
        self.classifier = nn.Linear(64, len(concepts))


def records():
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]


def train():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    rows = records()
    concepts = sorted({c for r in rows if r["split"] == "train" for c in r["labels"]["concepts"]})
    x = torch.stack([features(passage(r)) for r in rows])
    y = torch.tensor([[c in r["labels"]["concepts"] for c in concepts] for r in rows]).float()
    splits = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train","validation","test")}
    model = CourseEncoder(concepts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    best, weights = float("inf"), None
    for _ in range(100):
        logits = model.classifier(model.encoder(x[splits["train"]]))
        loss = F.binary_cross_entropy_with_logits(logits, y[splits["train"]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            valid = splits["validation"]
            score = F.binary_cross_entropy_with_logits(model.classifier(model.encoder(x[valid])), y[valid]).item()
            if score < best:
                best = score
                weights = {k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(weights)
    test = splits["test"]
    with torch.no_grad():
        predicted = model.classifier(model.encoder(x[test])).argmax(1)
        hits = int(y[test].gather(1,predicted[:,None]).sum())
    artifact = {"weights": weights, "concepts": concepts, "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
                "test_top_concept_hits": hits, "test_examples": len(test)}
    torch.save(artifact, WEIGHTS)
    print(f"Held-out top concept: {hits}/{len(test)}")


@torch.inference_mode()
def study(query):
    artifact = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    if hashlib.sha256(CORPUS.read_bytes()).hexdigest() != artifact["corpus_sha256"]:
        raise ValueError("Course checkpoint and evidence corpus do not match")
    rows = records()
    model = CourseEncoder(artifact["concepts"])
    model.load_state_dict(artifact["weights"])
    inputs = torch.stack([features(passage(r)) for r in rows])
    q = features(query)
    scores = F.cosine_similarity(model.encoder(inputs), model.encoder(q)[None,:]) + inputs @ q
    return [{"passage": passage(rows[i]), "source": rows[i]["source"], "tier": rows[i]["labels"]["knowledge_tier"]}
            for i in scores.topk(3).indices.tolist()]


if __name__ == "__main__":
    train()
