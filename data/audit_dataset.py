"""
audit_dataset.py — Stage 0 dataset audit and index builder.

Two jobs, both surfacing real issues found during Stage 0:

  1. status-vs-params conflict audit.
     status.json and params.json BOTH carry valid_geometry / valid_mesh /
     valid_named_selections, and they DISAGREE (e.g. variant 0001: status says
     valid_mesh=true, params says false). Downstream "is this sample usable?" logic
     must not pick silently. We report the conflict rate and apply an explicit rule.

  2. dataset_index.csv builder.
     One row per variant with paths, counts, validity (resolved), and whether the
     boundary named-selections needed for physics training are present.

Reads metadata (small JSON/CSV) DIRECTLY from the dataset zip so all 300 variants can
be audited without extracting 2.4 GB. Heavy mesh parsing is intentionally NOT done here
(that is per-variant Stage-1 work); we only check declared node/element counts and the
presence of the boundary selections.

Resolution rule for the status/params conflict (documented, not hidden):
  status.json is the SOURCE OF TRUTH; params.json carries stale generation-time values.
  Evidence (audit of all 300):
    - the conflict is 100% one-directional: status=True / params=False in all 229
      cases, never the reverse;
    - all 300 have status.valid_* = True and failed_elements_count = 0;
    - the dataset is the curated "valid_300" set and the manifest treats all 300 as
      valid; params.json also still says solved=false (a never-updated gen-time field).
  Therefore a field is resolved to its status.json value. (An earlier strict-AND rule
  was wrong: it discarded 76% of a dataset that is, by construction, fully valid.)
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

VALID_FIELDS = ("valid_geometry", "valid_mesh", "valid_named_selections")
REQUIRED_SELECTIONS = ("FIXED_TOP_FACES", "MOVE_INNER_RING_FACE")


@dataclass
class VariantAudit:
    variant_id: str
    subset: str                 # STRICT | COMPLEX (derived from variant_id prefix)
    sample_group: str
    source_matrix: str
    num_nodes: int
    num_elements: int
    failed_elements_count: int
    # per-field: agreement and resolved value
    status_valid_geometry: bool
    params_valid_geometry: bool
    status_valid_mesh: bool
    params_valid_mesh: bool
    status_valid_named_selections: bool
    params_valid_named_selections: bool
    resolved_valid_geometry: bool
    resolved_valid_mesh: bool
    resolved_valid_named_selections: bool
    has_fixed_faces: bool
    has_move_face: bool
    n_conflicts: int            # how many of the 3 valid_* fields disagree
    valid_for_physics: bool     # final gate


def _read_json_member(zf: zipfile.ZipFile, name: str) -> dict | None:
    try:
        with zf.open(name) as fh:
            return json.load(io.TextIOWrapper(fh, encoding="utf-8"))
    except KeyError:
        return None


def _manifest_rows(zf: zipfile.ZipFile) -> list[dict]:
    name = "packages/vcm_spring_valid_300_manifest.csv"
    with zf.open(name) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8-sig")  # strip BOM
        return list(csv.DictReader(text))


def _variant_member_prefix(zf: zipfile.ZipFile, variant_id: str) -> str | None:
    """Locate the in-zip directory prefix for a variant id."""
    needle = f"variants/{variant_id}/"
    for n in zf.namelist():
        if n.startswith(needle):
            return needle
    return None


def audit_variant(zf: zipfile.ZipFile, row: dict) -> VariantAudit:
    vid = row["variant_id"]
    prefix = _variant_member_prefix(zf, vid)
    status = _read_json_member(zf, prefix + "status.json") if prefix else None
    params = _read_json_member(zf, prefix + "params.json") if prefix else None
    ns = _read_json_member(zf, prefix + "named_selections.json") if prefix else None
    status = status or {}
    params = params or {}
    ns = ns or {}

    def sget(f):  # status value, default False
        return bool(status.get(f, False))

    def pget(f):  # params value, default False
        return bool(params.get(f, False))

    # status.json is source of truth (see module docstring); params is stale.
    resolved = {f: sget(f) for f in VALID_FIELDS}
    n_conflicts = sum(sget(f) != pget(f) for f in VALID_FIELDS)

    has_fixed = bool(ns.get("FIXED_TOP_FACES"))
    has_move = bool(ns.get("MOVE_INNER_RING_FACE"))

    valid_for_physics = (
        all(resolved.values())
        and has_fixed
        and has_move
        and int(row.get("failed_elements_count", "0") or 0) == 0
    )

    if "STRICT" in vid:
        subset = "STRICT"
    elif "COMPLEX" in vid:
        subset = "COMPLEX"
    else:
        subset = "UNKNOWN"

    return VariantAudit(
        variant_id=vid,
        subset=subset,
        sample_group=row.get("sample_group", ""),
        source_matrix=row.get("source_matrix", ""),
        num_nodes=int(row.get("num_nodes", 0) or 0),
        num_elements=int(row.get("num_elements", 0) or 0),
        failed_elements_count=int(row.get("failed_elements_count", 0) or 0),
        status_valid_geometry=sget("valid_geometry"),
        params_valid_geometry=pget("valid_geometry"),
        status_valid_mesh=sget("valid_mesh"),
        params_valid_mesh=pget("valid_mesh"),
        status_valid_named_selections=sget("valid_named_selections"),
        params_valid_named_selections=pget("valid_named_selections"),
        resolved_valid_geometry=resolved["valid_geometry"],
        resolved_valid_mesh=resolved["valid_mesh"],
        resolved_valid_named_selections=resolved["valid_named_selections"],
        has_fixed_faces=has_fixed,
        has_move_face=has_move,
        n_conflicts=n_conflicts,
        valid_for_physics=valid_for_physics,
    )


def run_audit(zip_path: str | Path, out_dir: str | Path = "reports") -> dict:
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        rows = _manifest_rows(zf)
        audits = [audit_variant(zf, r) for r in rows]

    # write dataset_index.csv
    index_csv = out_dir / "dataset_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(audits[0]).keys()))
        w.writeheader()
        for a in audits:
            w.writerow(asdict(a))

    # aggregate stats
    n = len(audits)
    n_conflict_any = sum(a.n_conflicts > 0 for a in audits)
    conflict_by_field = {
        f: sum(getattr(a, f"status_{f}") != getattr(a, f"params_{f}") for a in audits)
        for f in VALID_FIELDS
    }
    n_valid = sum(a.valid_for_physics for a in audits)
    n_missing_sel = sum(not (a.has_fixed_faces and a.has_move_face) for a in audits)
    by_group: dict[str, int] = {}
    for a in audits:
        by_group[a.sample_group] = by_group.get(a.sample_group, 0) + 1
    by_subset: dict[str, int] = {}
    for a in audits:
        by_subset[a.subset] = by_subset.get(a.subset, 0) + 1

    summary = {
        "n_variants": n,
        "n_with_any_conflict": n_conflict_any,
        "conflict_rate": round(n_conflict_any / n, 4) if n else 0,
        "conflicts_by_field": conflict_by_field,
        "n_valid_for_physics": n_valid,
        "n_missing_boundary_selection": n_missing_sel,
        "by_subset": by_subset,
        "by_sample_group": by_group,
    }

    # write audit report (markdown)
    report = out_dir / "dataset_audit_report.md"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# Dataset Audit Report (Stage 0)\n\n")
        fh.write(f"- variants audited: **{n}**\n")
        fh.write(f"- variants with >=1 status/params conflict: **{n_conflict_any}** "
                 f"({summary['conflict_rate']*100:.1f}%)\n")
        fh.write("- conflicts by field (status != params):\n")
        for f, c in conflict_by_field.items():
            fh.write(f"    - `{f}`: {c}\n")
        fh.write(f"- variants valid_for_physics (resolved AND + boundary present + 0 failed elems): "
                 f"**{n_valid}**\n")
        fh.write(f"- variants missing FIXED/MOVE selection: **{n_missing_sel}**\n\n")
        fh.write("## Resolution rule\n\n")
        fh.write("`status.json` is the **source of truth**; `params.json` carries stale "
                 "generation-time values. The conflict is 100% one-directional "
                 "(status=True / params=False), all 300 have status valid_*=True and 0 "
                 "failed elements, so each `valid_*` resolves to its status.json value.\n\n")
        fh.write("## By sample group\n\n")
        for g, c in sorted(by_group.items()):
            fh.write(f"- {g}: {c}\n")
        fh.write("\n## Index\n\nSee `dataset_index.csv` (one row per variant).\n")

    return {"summary": summary, "index_csv": str(index_csv), "report": str(report)}


if __name__ == "__main__":
    import sys

    zp = sys.argv[1] if len(sys.argv) > 1 else "../vcm_spring_valid_300_variants.zip"
    result = run_audit(zp)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nindex  -> {result['index_csv']}")
    print(f"report -> {result['report']}")
