import streamlit as st
from .model_cache import cached_model_path

QA_MODEL = "deepset/xlm-roberta-base-squad2"


@st.cache_resource
def load_qa_model(model_name: str = QA_MODEL):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
    path = cached_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)


def extract_answer(model, question: str, context: str, minimum_score: float = 0.15):
    result = model(question=question, context=context, handle_impossible_answer=True)
    start, end = result.get("start", -1), result.get("end", -1)
    if not (0 <= start < end <= len(context)) or result.get("score", 0) < minimum_score:
        return None
    answer = context[start:end]
    if answer != result.get("answer"):
        return None
    return dict(answer=answer, start=start, end=end, score=float(result["score"]))
