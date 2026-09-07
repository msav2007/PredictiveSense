"""Config: profiles load, bad input fails loudly, resolved config is serialisable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from predictivesense.config.settings import ConfigError, available_profiles, load_config
from predictivesense.core.enums import Mode

pytestmark = pytest.mark.unit


def test_both_shipped_profiles_load_and_validate(profiles_dir: Path) -> None:
    assert set(available_profiles(profiles_dir)) >= {"dev", "eval"}

    dev = load_config("dev")
    ev = load_config("eval")

    assert dev.profile == "dev"
    assert dev.mode is Mode.REALTIME
    assert ev.mode is Mode.RECORDED
    assert dev.source.target_fps > 0
    assert ev.consumer.sample_rate_hz > 0


def test_unknown_profile_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config("does-not-exist", profiles_dir=tmp_path)


def test_unknown_field_raises(write_profile, valid_profile_text: str) -> None:
    directory = write_profile("bad_field", valid_profile_text + "\nmystery_field: 1\n")
    with pytest.raises(ValidationError):
        load_config("bad_field", profiles_dir=directory)


def test_invalid_mode_raises(write_profile, valid_profile_text: str) -> None:
    text = valid_profile_text.replace("mode: realtime", "mode: invalid")
    directory = write_profile("bad_mode", text)
    with pytest.raises(ValidationError):
        load_config("bad_mode", profiles_dir=directory)


def test_out_of_range_value_raises(write_profile, valid_profile_text: str) -> None:
    text = valid_profile_text.replace("target_fps: 1000.0", "target_fps: -3.0")
    directory = write_profile("bad_value", text)
    with pytest.raises(ValidationError):
        load_config("bad_value", profiles_dir=directory)


def test_resolved_config_is_json_serialisable() -> None:
    payload = load_config("dev").as_json_dict()
    text = json.dumps(payload)
    round_tripped = json.loads(text)
    assert round_tripped["mode"] == "realtime"
    assert round_tripped["source"]["kind"] == "synthetic"
    assert "results_dir" in round_tripped
