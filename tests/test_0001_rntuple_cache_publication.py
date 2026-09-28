# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from uproot.models.RNTuple import Model_ROOT_3a3a_RNTuple, PageLink


class InterleavedRecords:
    def __init__(self, records, *, fail=False):
        self.records = records
        self.fail = fail
        self.paused = Event()
        self.resume = Event()
        self.lock = Lock()
        self.calls = 0

    def __iter__(self):
        with self.lock:
            self.calls += 1
            first = self.calls == 1
        yield self.records[0]
        if first:
            if self.fail:
                raise OSError("interrupted metadata read")
            self.paused.set()
            assert self.resume.wait(5), "metadata reader was not released"
        yield self.records[1]


def metadata_model(property_name, records, monkeypatch):
    model = Model_ROOT_3a3a_RNTuple.empty()
    model._page_list_envelopes = []
    model._cluster_summaries = None
    model._page_link_list = None
    if property_name == "page_list_envelopes":
        model._footer = SimpleNamespace(cluster_group_records=records)
        monkeypatch.setattr(model, "read_locator", lambda loc, size: (loc, None))
        monkeypatch.setattr(PageLink, "read", lambda self, chunk, cursor, ctx: chunk)
    else:
        model._page_list_envelopes = records
    return model


def records_for(property_name):
    if property_name == "page_list_envelopes":
        return [
            SimpleNamespace(
                page_list_link=SimpleNamespace(locator=i, env_uncomp_size=0)
            )
            for i in range(2)
        ]
    attribute = "pagelinklist" if property_name == "page_link_list" else property_name
    return [SimpleNamespace(**{attribute: [i]}) for i in range(2)]


@pytest.mark.parametrize(
    "property_name", ["page_link_list", "cluster_summaries", "page_list_envelopes"]
)
def test_concurrent_readers_receive_complete_metadata(property_name, monkeypatch):
    # Pause the first reader between records and read from another thread while
    # it is paused. This reproduces partial cache publication with or without
    # the GIL, without relying on scheduler timing or repeated stress loops.
    records = InterleavedRecords(records_for(property_name))
    model = metadata_model(property_name, records, monkeypatch)

    def snapshot():
        return list(getattr(model, property_name))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(snapshot)
        try:
            assert records.paused.wait(5), "first reader did not reach the pause"
            second = pool.submit(snapshot).result(timeout=5)
        finally:
            records.resume.set()
        assert first.result(timeout=5) == [0, 1]
    assert second == [0, 1]
    assert snapshot() == [0, 1]


@pytest.mark.parametrize(
    "property_name", ["page_link_list", "cluster_summaries", "page_list_envelopes"]
)
def test_failed_metadata_read_can_retry(property_name, monkeypatch):
    records = InterleavedRecords(records_for(property_name), fail=True)
    model = metadata_model(property_name, records, monkeypatch)
    with pytest.raises(OSError, match="interrupted metadata read"):
        getattr(model, property_name)
    assert list(getattr(model, property_name)) == [0, 1]
