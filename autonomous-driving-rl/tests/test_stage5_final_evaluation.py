from cloud.kaggle.final_evaluation import summarize


def test_localization_failure_is_reported_without_counting_it_as_a_safety_outcome():
    rows = [
        dict(clean=True, outcome="SUCCESS", technical_failure=False, progress=0.99,
             speed_km_h=35.0, success_seconds=42.0),
        dict(clean=False, outcome="CRASH_VEHICLE", technical_failure=False, progress=0.5,
             speed_km_h=25.0, success_seconds=None),
        dict(clean=False, outcome="TIMEOUT", technical_failure=False, progress=0.1,
             speed_km_h=0.0, success_seconds=None),
        dict(clean=False, outcome="TASK_CONTEXT_LOCALIZATION", technical_failure=True, progress=None,
             speed_km_h=None, success_seconds=None),
    ]
    result = summarize({"cases": rows})
    assert result["cases"] == 4
    assert result["clean_successes"] == 1
    assert result["safety_failures"] == result["timeouts"] == result["technical_failures"] == 1
    assert result["mean_success_seconds"] == 42.0
    assert result["mean_speed_km_h"] == 20.0
