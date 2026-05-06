"""WuDaoZi test suite."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestToolResult:
    def test_success_result(self):
        from wudaozi.api import ToolResult

        r = ToolResult(success=True, data={"path": "test.png"}, metadata={"v": "1.0.0"})
        assert r.success is True
        assert r.data == {"path": "test.png"}
        assert r.error is None

    def test_failure_result(self):
        from wudaozi.api import ToolResult

        r = ToolResult(success=False, error="model not found")
        assert r.success is False
        assert r.error == "model not found"
        assert r.data is None

    def test_to_dict(self):
        from wudaozi.api import ToolResult

        r = ToolResult(success=True, data=[1, 2], metadata={"x": 1})
        d = r.to_dict()
        assert set(d.keys()) == {"success", "data", "error", "metadata"}
        assert d["success"] is True
        assert d["data"] == [1, 2]

    def test_default_metadata_isolation(self):
        from wudaozi.api import ToolResult

        r1 = ToolResult(success=True)
        r2 = ToolResult(success=True)
        r1.metadata["a"] = 1
        assert "a" not in r2.metadata


class TestGenerateImageAPI:
    @patch("wudaozi.core.WuDaoZiEngine")
    def test_empty_prompt(self, mock_engine_cls):
        from wudaozi.api import generate_image

        result = generate_image(prompt="")
        assert result.success is False
        assert "empty" in result.error.lower()

    @patch("wudaozi.core.WuDaoZiEngine")
    def test_valid_prompt(self, mock_engine_cls):
        mock_engine = MagicMock()
        mock_img = MagicMock()
        mock_engine.generate.return_value = mock_img
        mock_engine_cls.return_value = mock_engine
        from wudaozi.api import generate_image

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_image(prompt="A mountain landscape", output=str(Path(tmpdir) / "test.png"))
            assert result.success is True
            assert "version" in result.metadata

    @patch("wudaozi.core.WuDaoZiEngine")
    def test_with_reference(self, mock_engine_cls):
        mock_engine = MagicMock()
        mock_img = MagicMock()
        mock_engine.generate.return_value = mock_img
        mock_engine_cls.return_value = mock_engine
        from wudaozi.api import generate_image

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_image(
                prompt="A dragon",
                output=str(Path(tmpdir) / "test.png"),
                reference_style="ink wash painting",
                reference_character="a warrior in armor",
            )
            assert result.success is True


class TestBatchGenerateAPI:
    @patch("wudaozi.core.WuDaoZiEngine")
    def test_empty_prompts_file(self, mock_engine_cls):
        from wudaozi.api import batch_generate

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.txt"
            empty_file.write_text("")
            result = batch_generate(prompts_file=str(empty_file))
            assert result.success is False

    @patch("wudaozi.core.WuDaoZiEngine")
    def test_missing_prompts_file(self, mock_engine_cls):
        from wudaozi.api import batch_generate

        result = batch_generate(prompts_file="/nonexistent/file.txt")
        assert result.success is False
        assert "not found" in result.error.lower()


class TestLoadReferenceAPI:
    def test_create_reference(self):
        from wudaozi.api import load_reference

        result = load_reference(name="Hero", style="ink wash", character="warrior in red armor")
        assert result.success is True
        assert result.data["name"] == "Hero"
        assert result.data["style"] == "ink wash"

    def test_missing_name(self):
        from wudaozi.api import load_reference

        result = load_reference(name="", style="test")
        assert result.success is False

    def test_load_from_json_file(self):
        from wudaozi.api import load_reference

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_file = Path(tmpdir) / "ref.json"
            ref_file.write_text(
                json.dumps(
                    {
                        "name": "Muse",
                        "style": "oil painting",
                        "character": "lady in white",
                        "reference_images": ["img1.png"],
                    }
                )
            )
            result = load_reference(name="test", from_file=str(ref_file))
            assert result.success is True
            assert result.data["name"] == "Muse"
            assert result.data["style"] == "oil painting"


class TestCreateSeriesAPI:
    @patch("wudaozi.core.WuDaoZiEngine")
    def test_empty_name(self, mock_engine_cls):
        from wudaozi.api import create_series

        result = create_series(name="", theme="test", prompts_file="test.txt")
        assert result.success is False
        assert "empty" in result.error.lower()


class TestToolsSchema:
    def test_tools_is_list(self):
        from wudaozi.tools import TOOLS

        assert isinstance(TOOLS, list)
        assert len(TOOLS) >= 4

    def test_tool_names(self):
        from wudaozi.tools import TOOLS

        for tool in TOOLS:
            name = tool["function"]["name"]
            assert name.startswith("wudaozi_")

    def test_tool_structure(self):
        from wudaozi.tools import TOOLS

        for tool in TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "properties" in func["parameters"]
            assert "required" in func["parameters"]

    def test_required_fields_in_properties(self):
        from wudaozi.tools import TOOLS

        for tool in TOOLS:
            func = tool["function"]
            props = func["parameters"]["properties"]
            for req in func["parameters"]["required"]:
                assert req in props, f"Required '{req}' not in properties"


class TestToolsDispatch:
    def test_dispatch_unknown_tool(self):
        from wudaozi.tools import dispatch

        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("nonexistent_tool", {})

    def test_dispatch_json_string_args(self):
        from wudaozi.tools import dispatch

        args = json.dumps({"name": "Test", "style": "test style"})
        result = dispatch("wudaozi_load_reference", args)
        assert isinstance(result, dict)
        assert "success" in result

    def test_dispatch_load_reference(self):
        from wudaozi.tools import dispatch

        result = dispatch("wudaozi_load_reference", {"name": "Hero", "style": "ink wash"})
        assert result["success"] is True
        assert result["data"]["name"] == "Hero"

    def test_dispatch_generate_empty_prompt(self):
        from wudaozi.tools import dispatch

        result = dispatch("wudaozi_generate_image", {"prompt": ""})
        assert result["success"] is False


class TestCLIFlags:
    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "wudaozi"] + list(args),
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_version_flag(self):
        r = self._run_cli("-V")
        assert r.returncode == 0
        assert "wudaozi" in r.stdout.lower()

    def test_help_has_unified_flags(self):
        r = self._run_cli("--help")
        assert r.returncode == 0
        assert "--json" in r.stdout
        assert "-q" in r.stdout or "--quiet" in r.stdout
        assert "-v" in r.stdout or "--verbose" in r.stdout

    def test_generate_help(self):
        r = self._run_cli("generate", "--help")
        assert r.returncode == 0
        assert "--prompt" in r.stdout

    def test_batch_help(self):
        r = self._run_cli("batch", "--help")
        assert r.returncode == 0
        assert "--prompts" in r.stdout

    def test_series_help(self):
        r = self._run_cli("series", "--help")
        assert r.returncode == 0
        assert "--name" in r.stdout


class TestPackageExports:
    def test_version(self):
        import wudaozi

        assert hasattr(wudaozi, "__version__")
        assert isinstance(wudaozi.__version__, str)

    def test_toolresult(self):
        from wudaozi import ToolResult

        assert callable(ToolResult)

    def test_generate_image_exported(self):
        from wudaozi import generate_image

        assert callable(generate_image)

    def test_batch_generate_exported(self):
        from wudaozi import batch_generate

        assert callable(batch_generate)

    def test_load_reference_exported(self):
        from wudaozi import load_reference

        assert callable(load_reference)

    def test_create_series_exported(self):
        from wudaozi import create_series

        assert callable(create_series)

    def test_all_defined(self):
        import wudaozi

        assert hasattr(wudaozi, "__all__")
