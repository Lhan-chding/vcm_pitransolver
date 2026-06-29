"""Tests for data/audit_dataset.py — conflict resolution + index building."""

from __future__ import annotations

import json
import zipfile

import audit_dataset as ad


def _make_zip(tmp_path, variants):
    """Build a minimal dataset zip with manifest + per-variant status/params/ns.

    variants: list of dicts with keys:
      id, status (dict), params (dict), ns (dict), failed (int)
    """
    zp = tmp_path / "ds.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        # manifest
        lines = ["variant_id,source_matrix,path_to_variant,sample_group,"
                 "variables_changed,num_nodes,num_elements,failed_elements_count"]
        for v in variants:
            lines.append(
                f"{v['id']},m.csv,/p/{v['id']},grp,x,1000,200,{v.get('failed', 0)}"
            )
        zf.writestr("packages/vcm_spring_valid_300_manifest.csv", "\n".join(lines) + "\n")
        for v in variants:
            pre = f"variants/{v['id']}/"
            zf.writestr(pre + "status.json", json.dumps(v["status"]))
            zf.writestr(pre + "params.json", json.dumps(v["params"]))
            zf.writestr(pre + "named_selections.json", json.dumps(v["ns"]))
    return zp


def _full_ns():
    return {"FIXED_TOP_FACES": [1, 2, 3, 4], "MOVE_INNER_RING_FACE": [99]}


def test_status_is_source_of_truth(tmp_path):
    # status says valid, params (stale) says invalid -> resolved should follow status
    variants = [
        {
            "id": "VCM_STRICT_0001",
            "status": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "params": {"valid_geometry": True, "valid_mesh": False,
                       "valid_named_selections": False},
            "ns": _full_ns(),
        }
    ]
    zp = _make_zip(tmp_path, variants)
    with zipfile.ZipFile(zp) as zf:
        rows = ad._manifest_rows(zf)
        a = ad.audit_variant(zf, rows[0])
    assert a.resolved_valid_mesh is True            # follows status, not params
    assert a.resolved_valid_named_selections is True
    assert a.n_conflicts == 2                        # mesh + named_selections disagree
    assert a.valid_for_physics is True
    assert a.subset == "STRICT"


def test_genuinely_invalid_is_dropped(tmp_path):
    # status itself says invalid mesh -> must NOT be valid_for_physics
    variants = [
        {
            "id": "VCM_COMPLEX_0002",
            "status": {"valid_geometry": True, "valid_mesh": False,
                       "valid_named_selections": True},
            "params": {"valid_geometry": True, "valid_mesh": False,
                       "valid_named_selections": True},
            "ns": _full_ns(),
        }
    ]
    zp = _make_zip(tmp_path, variants)
    with zipfile.ZipFile(zp) as zf:
        rows = ad._manifest_rows(zf)
        a = ad.audit_variant(zf, rows[0])
    assert a.resolved_valid_mesh is False
    assert a.valid_for_physics is False
    assert a.subset == "COMPLEX"


def test_missing_boundary_selection_drops_sample(tmp_path):
    variants = [
        {
            "id": "VCM_STRICT_0003",
            "status": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "params": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "ns": {"FIXED_TOP_FACES": [1, 2, 3, 4]},  # no MOVE face
        }
    ]
    zp = _make_zip(tmp_path, variants)
    with zipfile.ZipFile(zp) as zf:
        rows = ad._manifest_rows(zf)
        a = ad.audit_variant(zf, rows[0])
    assert a.has_move_face is False
    assert a.valid_for_physics is False


def test_failed_elements_drops_sample(tmp_path):
    variants = [
        {
            "id": "VCM_STRICT_0004",
            "status": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "params": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "ns": _full_ns(),
            "failed": 5,
        }
    ]
    zp = _make_zip(tmp_path, variants)
    with zipfile.ZipFile(zp) as zf:
        rows = ad._manifest_rows(zf)
        a = ad.audit_variant(zf, rows[0])
    assert a.valid_for_physics is False


def test_run_audit_writes_index_and_report(tmp_path):
    variants = [
        {
            "id": "VCM_STRICT_0005",
            "status": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "params": {"valid_geometry": True, "valid_mesh": False,
                       "valid_named_selections": False},
            "ns": _full_ns(),
        },
        {
            "id": "VCM_COMPLEX_0006",
            "status": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "params": {"valid_geometry": True, "valid_mesh": True,
                       "valid_named_selections": True},
            "ns": _full_ns(),
        },
    ]
    zp = _make_zip(tmp_path, variants)
    out = tmp_path / "rep"
    result = ad.run_audit(zp, out_dir=out)
    s = result["summary"]
    assert s["n_variants"] == 2
    assert s["n_with_any_conflict"] == 1
    assert s["n_valid_for_physics"] == 2
    assert s["by_subset"] == {"STRICT": 1, "COMPLEX": 1}
    assert (out / "dataset_index.csv").exists()
    assert (out / "dataset_audit_report.md").exists()
