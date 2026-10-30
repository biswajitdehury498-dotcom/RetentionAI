import pytest

from src.retentionai_service import RetentionAIAssistant


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Show me critical-risk customers", "critical_risk"),
        ("Find high-value high-risk customers", "high_value_high_risk"),
        ("Who are the highest risk customers?", "highest_risk_customers"),
        ("Show campaign priorities", "campaign_priorities"),
        ("Explain the risk segments", "risk_segments"),
        ("Tell me a joke", None),
    ],
)
def test_route_question(question, expected):
    assert RetentionAIAssistant.route_question(question) == expected


@pytest.mark.parametrize("question", ["", "   ", None, 123])
def test_route_question_rejects_invalid_input(question):
    with pytest.raises(ValueError):
        RetentionAIAssistant.route_question(question)
