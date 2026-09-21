import json
import runpy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.aiops_pipeline import run_pipeline
from src.anomaly_detector import AnomalyDetector
from src.calculations import area_of_circle, get_nth_fibonacci
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_anomaly_detector_reports_each_signal():
    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "response_time_ms": 501,
        "cpu_percent": 81,
        "memory_percent": 81,
        "log_level": "WARNING",
    }

    event = AnomalyDetector().detect(record)

    assert event["reasons"] == [
        "High response time",
        "High CPU utilization",
        "High memory utilization",
        "Error log detected",
    ]


def test_producer_rejects_empty_event():
    topic = EventTopic("anomaly-events")

    assert EventProducer(topic).publish(None) is False
    assert topic.get_messages() == []


def test_topic_clear_removes_messages():
    topic = EventTopic("anomaly-events")
    topic.publish({"type": "ANOMALY"})

    topic.clear()

    assert topic.get_messages() == []


def test_calculations_reject_negative_values_and_calculate_sequence():
    with pytest.raises(ValueError):
        area_of_circle(-1)

    with pytest.raises(ValueError):
        get_nth_fibonacci(-1)

    assert get_nth_fibonacci(10) == 55


def test_pipeline_loads_data_and_detects_anomalies(tmp_path):
    data_file = tmp_path / "service_data.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-09-20T10:00:00",
                    "service": "payment-service",
                    "response_time_ms": 700,
                    "cpu_percent": 40,
                    "memory_percent": 40,
                    "log_level": "ERROR",
                },
                {
                    "timestamp": "2026-09-20T10:01:00",
                    "service": "payment-service",
                    "response_time_ms": 100,
                    "cpu_percent": 40,
                    "memory_percent": 40,
                    "log_level": "INFO",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = run_pipeline(data_file)

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert result["events_consumed"] == []


def test_pipeline_cli_entrypoint(capsys, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    runpy.run_path("src/aiops_pipeline.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "AIOps Pipeline Result" in output
    assert "Records processed: 10" in output