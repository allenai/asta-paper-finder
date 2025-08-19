import pytest
from pydantic import ValidationError

from mabool.agents.query_analyzer.query_analyzer_prompts import IdentifyRelevanceCriteriaOutput, RelevanceCriterion


def test_identify_relevance_criteria_output_valid() -> None:
    criteria = [
        RelevanceCriterion(name="criterion1", description="desc1", weight=0.5),
        RelevanceCriterion(name="criterion2", description="desc2", weight=0.5),
    ]
    output = IdentifyRelevanceCriteriaOutput(required_relevance_critieria=criteria)
    assert output.required_relevance_critieria == criteria
    assert output.nice_to_have_relevance_criteria is None
    assert output.clarification_questions is None


def test_identify_relevance_criteria_output_no_required_or_clarification() -> None:
    with pytest.raises(ValidationError):
        IdentifyRelevanceCriteriaOutput()


def test_identify_relevance_criteria_output_duplicate_criterion_names() -> None:
    criteria = [
        RelevanceCriterion(name="criterion1", description="desc1", weight=0.5),
        RelevanceCriterion(name="criterion1", description="desc2", weight=0.5),
    ]
    with pytest.raises(ValidationError):
        IdentifyRelevanceCriteriaOutput(required_relevance_critieria=criteria)

    with pytest.raises(ValidationError):
        IdentifyRelevanceCriteriaOutput(
            required_relevance_critieria=criteria[:1], nice_to_have_relevance_criteria=criteria[1:]
        )


def test_identify_relevance_criteria_output_sum_to_one() -> None:
    criteria = [
        RelevanceCriterion(name="criterion1", description="desc1", weight=0.5),
        RelevanceCriterion(name="criterion2", description="desc2", weight=0.6),
    ]
    with pytest.raises(ValidationError):
        IdentifyRelevanceCriteriaOutput(required_relevance_critieria=criteria)


def test_identify_relevance_criteria_output_with_clarification_questions() -> None:
    questions = ["What is the specific topic?", "Can you provide more details?"]
    output = IdentifyRelevanceCriteriaOutput(clarification_questions=questions)
    assert output.required_relevance_critieria is None
    assert output.nice_to_have_relevance_criteria is None
    assert output.clarification_questions == questions


def test_identify_relevance_criteria_output_with_nice_to_have_criteria() -> None:
    required_criteria = [
        RelevanceCriterion(name="criterion1", description="desc1", weight=0.5),
        RelevanceCriterion(name="criterion2", description="desc2", weight=0.5),
    ]
    nice_to_have_criteria = [
        RelevanceCriterion(name="criterion3", description="desc3", weight=0.3),
        RelevanceCriterion(name="criterion4", description="desc4", weight=0.7),
    ]
    output = IdentifyRelevanceCriteriaOutput(
        required_relevance_critieria=required_criteria, nice_to_have_relevance_criteria=nice_to_have_criteria
    )
    assert output.required_relevance_critieria == required_criteria
    assert output.nice_to_have_relevance_criteria == nice_to_have_criteria
    assert output.clarification_questions is None
