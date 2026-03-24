import pytest
from pydantic import ValidationError

from models import AskRequest, EvalGenRequest, EvalRunRequest, QueryRequest


def test_query_request_default_top_k():
    req = QueryRequest(query="neural networks")
    assert req.top_k == 5


def test_ask_request_default_top_k():
    req = AskRequest(query="what is attention?")
    assert req.top_k == 8


def test_eval_gen_request_default_num_questions():
    req = EvalGenRequest()
    assert req.num_questions == 5


def test_eval_run_request_default_k():
    req = EvalRunRequest()
    assert req.k == 5


def test_query_request_requires_query_field():
    with pytest.raises(ValidationError):
        QueryRequest()


def test_ask_request_custom_top_k():
    req = AskRequest(query="test", top_k=12)
    assert req.top_k == 12


def test_eval_gen_request_custom_num_questions():
    req = EvalGenRequest(num_questions=20)
    assert req.num_questions == 20
