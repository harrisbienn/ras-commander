"""Preserve native coupling diagnostics independently of completion status."""

from ras_commander.results.ResultsParser import ResultsParser


def test_coupling_errors_preserve_synthetic_time_and_repeat_occurrences():
    messages = (
        "01JAN3000 24:00:00    1D/2D Flow error\t -1.7532 \t OuachitaRv   10.6   92442\r\r\n"
        "01JAN3000 24:00:00    1D/2D Flow error\t 5. \t OuachitaRv   10.6   92442\n"
        "02JAN3000 13:00:30    1D/2D Flow error\t -9.1D+1 \t River Name   Reach Name   48477\n"
        "Complete Process\n"
    )
    result = ResultsParser.summarize_coupling_errors(messages)
    assert result["marker_line_count"] == result["parsed_line_count"] == 3
    assert result["unparsed_line_count"] == 0
    first, second = result["locations"]
    assert first["location"] == "OuachitaRv 10.6 92442"
    assert first["count"] == 2
    assert first["first_timestamp"] == "01JAN3000 24:00:00"
    assert first["minimum_reported_error"] == -1.7532
    assert first["maximum_absolute_reported_error"] == 5.0
    assert second["maximum_absolute_reported_error"] == 91.0
    assert second["location"] == "River Name Reach Name 48477"
    assert result["reported_error_units"] == "unspecified-native"
    assert ResultsParser.is_successful_completion(messages) is False


def test_unparsed_and_nonfinite_errors_are_not_silently_dropped():
    result = ResultsParser.summarize_coupling_errors(
        "1D/2D Flow error unknown\n02JAN3000 13:00:30 1D/2D Flow error 1e999 River Reach 12\n"
    )
    assert result["marker_line_count"] == result["unparsed_line_count"] == 2
    assert result["parsed_line_count"] == 0
    assert len(result["unparsed_examples"]) == 2


def test_volume_accounting_metrics_are_not_coupling_errors():
    result = ResultsParser.summarize_coupling_errors(
        "Overall Volume Accounting Error in Acre Feet: 10\nComplete Process\n"
    )
    assert result["marker_line_count"] == 0
    assert result["locations"] == []


def test_event_times_normalize_midnight_and_preserve_order():
    messages = (
        "01JAN3000 24:00:00 1D/2D Flow error 5 River Reach 1\n"
        "02JAN3000 00:00:00 1D/2D Flow error 4 River Reach 1\n"
        "32JAN3000 24:00:00 1D/2D Flow error 2 River Reach 1\n"
    )
    result = ResultsParser.get_coupling_error_events(messages)
    assert [event["time"] for event in result["events"]] == ["3000-01-02T00:00:00"] * 2
    assert [event["reported_error"] for event in result["events"]] == [5, 4]
    assert result["unparsed_line_count"] == 1
    assert result["events"][0]["timestamp"] == "01JAN3000 24:00:00"
