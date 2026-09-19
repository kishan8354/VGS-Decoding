"""Dataset loaders.

VQA-RAD      : HF "flaviagiammarino/vqa-rad" test split (451 QA pairs) -- the split VASE uses.
               closed-ended := answer is yes/no. Paper reports 200 open + 251 closed; the loader
               prints the counts so you can confirm the split matches.
MIMIC-Diff-VQA: needs PhysioNet credentialed access (Medical-Diff-VQA + MIMIC-CXR-JPG).
               Column names / view selection below are the common layout; verify against your
               download and adjust the COLS dict if needed.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from PIL import Image

YES_NO = {"yes", "no"}


@dataclass
class Sample:
    qid: str
    question: str
    answer: str
    is_closed: bool
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    ref_image_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        if self.image is not None:
            return self.image
        return Image.open(self.image_path)


def load_vqa_rad(split: str = "test", limit: Optional[int] = None) -> List[Sample]:
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/vqa-rad", split=split)
    out = []
    for i, ex in enumerate(ds):
        ans = str(ex["answer"]).strip()
        out.append(Sample(qid=f"vqarad-{split}-{i:05d}", question=ex["question"], answer=ans,
                          is_closed=ans.lower() in YES_NO, image=ex["image"]))
    _report("VQA-RAD", out, expected=(200, 251) if split == "test" else None)
    return out[:limit] if limit else out


# ---------------------------------------------------------------- MIMIC-Diff-VQA
COLS = dict(subject="subject_id", study="study_id", ref="ref_id", qtype="question_type",
            question="question", answer="answer", split="split")
VIEW_PRIORITY = ["PA", "AP", "AP AXIAL", "PA LLD", "AP LLD", "LATERAL", "LL"]


def _study_to_jpg(metadata_csv: str, jpg_root: str):
    """study_id -> path of one frontal image (PA preferred) in MIMIC-CXR-JPG."""
    import pandas as pd
    md = pd.read_csv(metadata_csv, usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"])
    md["rank"] = md["ViewPosition"].map({v: i for i, v in enumerate(VIEW_PRIORITY)}).fillna(99)
    md = md.sort_values(["study_id", "rank"]).drop_duplicates("study_id")
    paths = {}
    for r in md.itertuples(index=False):
        sid, stid = str(int(r.subject_id)), str(int(r.study_id))
        paths[int(r.study_id)] = os.path.join(jpg_root, "files", f"p{sid[:2]}", f"p{sid}",
                                             f"s{stid}", f"{r.dicom_id}.jpg")
    return paths


def load_mimic_diff_vqa(qa_csv: str, metadata_csv: str, jpg_root: str, split: str = "test",
                        limit: Optional[int] = None, sample_n: Optional[int] = None,
                        seed: int = 0) -> List[Sample]:
    """qa_csv: mimic_pair_questions.csv from Medical-Diff-VQA.

    NOTE: the paper evaluates 13,121 test samples (8,380 open / 4,741 closed) but does not say
    how that subset was chosen. Use sample_n to draw a seeded subset and report how you chose it.
    """
    import pandas as pd
    df = pd.read_csv(qa_csv)
    df = df[df[COLS["split"]] == split].reset_index(drop=True)
    study2jpg = _study_to_jpg(metadata_csv, jpg_root)
    out = []
    for i, r in df.iterrows():
        main = study2jpg.get(int(r[COLS["study"]]))
        if main is None:
            continue
        ref = study2jpg.get(int(r[COLS["ref"]])) if COLS["ref"] in r and not pd.isna(r[COLS["ref"]]) else None
        ans = str(r[COLS["answer"]]).strip()
        out.append(Sample(qid=f"mimicdiff-{split}-{i:07d}", question=str(r[COLS["question"]]),
                          answer=ans, is_closed=ans.lower() in YES_NO, image_path=main,
                          ref_image_path=ref, meta={"question_type": r[COLS["qtype"]]}))
    if sample_n and sample_n < len(out):
        random.Random(seed).shuffle(out)
        out = sorted(out[:sample_n], key=lambda s: s.qid)
    _report("MIMIC-Diff-VQA", out, expected=(8380, 4741))
    return out[:limit] if limit else out


def _report(name, samples, expected=None):
    n_closed = sum(s.is_closed for s in samples)
    n_open = len(samples) - n_closed
    msg = f"[data] {name}: {len(samples)} samples | open={n_open} closed={n_closed}"
    if expected:
        msg += f" | paper: open={expected[0]} closed={expected[1]}"
    print(msg)


def iter_batches(samples, bs) -> Iterator[list]:
    for i in range(0, len(samples), bs):
        yield samples[i:i + bs]
