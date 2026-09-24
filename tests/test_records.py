"""Shared record helpers: the file lock and torn-line tolerance."""
import pytest

from memval.records import load_records, locked


def test_lock_blocks_second_holder(tmp_path):
    target = tmp_path / "results.jsonl"
    with locked(target):
        with pytest.raises(RuntimeError, match="another memval process"):
            with locked(target):
                pass
    # Released: a later acquirer proceeds.
    with locked(target):
        pass


def test_lock_file_is_a_sibling(tmp_path):
    target = tmp_path / "results.jsonl"
    with locked(target):
        assert (tmp_path / "results.jsonl.lock").exists()


def test_load_records_skips_torn_tail(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"task_id": "a"}\n{"task_id": "b"}\n{"task_id": "c", "con'
    )
    assert [r["task_id"] for r in load_records(path)] == ["a", "b"]
