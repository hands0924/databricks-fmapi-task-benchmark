"""benchmark.grade_tasks.render_and_capture — Playwright driving, fully faked.

Chromium is optional in this repo (the grader must keep working without it), so
these tests inject a fake `playwright.sync_api` module instead of rendering.
"""

import sys
import types

import pytest

from benchmark import grade_tasks


class FakeNode:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail

    def scroll_into_view_if_needed(self):
        pass

    def screenshot(self, path):
        if self.fail:
            raise RuntimeError("clip failed")
        open(path, "wb").write(b"png")


class FakePage:
    def __init__(self, nodes, section_nodes=(), console=(), goto_exc=None):
        self._nodes = list(nodes)
        self._sections = list(section_nodes)
        self._console = list(console)
        self._goto_exc = goto_exc
        self._handlers = {}
        self.full_page_shot = None

    def on(self, event, handler):
        self._handlers[event] = handler

    def goto(self, url, wait_until=None):
        if self._goto_exc:
            raise self._goto_exc
        self.url = url

    def wait_for_timeout(self, _ms):
        for msg in self._console:
            self._handlers["console"](msg)

    def query_selector_all(self, selector):
        return self._nodes if selector == ".slide" else self._sections

    def screenshot(self, path, full_page=False):
        self.full_page_shot = path
        open(path, "wb").write(b"png")


class FakeBrowser:
    def __init__(self, page):
        self.page = page
        self.closed = False

    def new_page(self, viewport=None):
        return self.page

    def close(self):
        self.closed = True


class FakeMessage:
    def __init__(self, text, type="error"):
        self.text = text
        self.type = type


def install_fake_playwright(monkeypatch, *, page=None, launch_exc=None):
    class FakeChromium:
        def launch(self):
            if launch_exc:
                raise launch_exc
            return FakeBrowser(page)

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    module = types.ModuleType("playwright.sync_api")
    module.sync_playwright = lambda: FakePlaywright()
    pkg = types.ModuleType("playwright")
    pkg.sync_api = module
    monkeypatch.setitem(sys.modules, "playwright", pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", module)


@pytest.fixture
def html(tmp_path):
    path = tmp_path / "slides.html"
    path.write_text("<!DOCTYPE html><html><body></body></html>", encoding="utf-8")
    return path


def test_render_screenshots_each_slide(html, tmp_path, monkeypatch):
    page = FakePage([FakeNode("a"), FakeNode("b")])
    install_fake_playwright(monkeypatch, page=page)

    out = grade_tasks.render_and_capture(html, tmp_path / "shots")

    assert out["rendered_ok"] is True
    assert out["console_errors"] == 0
    assert out["screenshot_paths"] == ["slide_01.png", "slide_02.png"]
    assert (tmp_path / "shots" / "slide_01.png").exists()
    assert page.url == html.resolve().as_uri()


def test_render_counts_console_and_page_errors(html, tmp_path, monkeypatch):
    page = FakePage([FakeNode("a")],
                    console=[FakeMessage("boom"), FakeMessage("warn", type="warning")])
    install_fake_playwright(monkeypatch, page=page)

    out = grade_tasks.render_and_capture(html, tmp_path / "shots")
    assert out["console_errors"] == 1  # warnings are not errors


def test_render_falls_back_to_reveal_sections(html, tmp_path, monkeypatch):
    page = FakePage([], section_nodes=[FakeNode("s1")])
    install_fake_playwright(monkeypatch, page=page)

    out = grade_tasks.render_and_capture(html, tmp_path / "shots")
    assert out["screenshot_paths"] == ["slide_01.png"]


def test_render_falls_back_to_full_page_when_no_slide_nodes(html, tmp_path, monkeypatch):
    page = FakePage([])
    install_fake_playwright(monkeypatch, page=page)

    out = grade_tasks.render_and_capture(html, tmp_path / "shots")
    assert out["screenshot_paths"] == ["slide_full.png"]
    assert page.full_page_shot is not None


def test_render_full_page_when_every_clip_fails(html, tmp_path, monkeypatch):
    page = FakePage([FakeNode("a", fail=True)])
    install_fake_playwright(monkeypatch, page=page)

    out = grade_tasks.render_and_capture(html, tmp_path / "shots")
    assert out["screenshot_paths"] == ["slide_full.png"]
    assert out["rendered_ok"] is True


def test_render_notes_chromium_launch_failure(html, tmp_path, monkeypatch):
    install_fake_playwright(monkeypatch, launch_exc=RuntimeError("no chromium"))

    out = grade_tasks.render_and_capture(html, tmp_path / "shots")
    assert out["rendered_ok"] is False
    assert "chromium launch failed" in out["render_note"]
    assert "playwright install chromium" in out["render_note"]


def test_render_notes_navigation_failure(html, tmp_path, monkeypatch):
    page = FakePage([], goto_exc=RuntimeError("nav died"))
    install_fake_playwright(monkeypatch, page=page)

    out = grade_tasks.render_and_capture(html, tmp_path / "shots")
    assert out["rendered_ok"] is False
    assert "render failed (RuntimeError: nav died)" in out["render_note"]
