"""Tests for the CLI module."""

import pytest
from typer.testing import CliRunner
from oxidizedvision.cli import app

runner = CliRunner()


class TestCLIConvert:
    def test_convert_command(self, simple_config_yaml):
        result = runner.invoke(app, ["convert", simple_config_yaml])
        assert result.exit_code == 0
        assert "Conversion complete" in result.stdout or "TorchScript" in result.stdout

    def test_convert_nonexistent_config(self):
        result = runner.invoke(app, ["convert", "nonexistent.yml"])
        assert result.exit_code == 1


class TestCLIValidate:
    def test_validate_command(self, simple_config_yaml, simple_config):
        # First convert
        runner.invoke(app, ["convert", simple_config_yaml])
        # Then validate
        result = runner.invoke(app, ["validate", simple_config_yaml])
        assert result.exit_code == 0


class TestCLIProfile:
    def test_profile_command(self, simple_config_yaml):
        result = runner.invoke(app, ["profile", simple_config_yaml])
        assert result.exit_code == 0
        assert "parameters" in result.stdout.lower() or "Model Profile" in result.stdout


class TestCLIList:
    def test_list_command(self):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0


class TestCLIHelp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "convert" in result.stdout
        assert "validate" in result.stdout
        assert "benchmark" in result.stdout
        assert "optimize" in result.stdout
        assert "profile" in result.stdout
        assert "package" in result.stdout
