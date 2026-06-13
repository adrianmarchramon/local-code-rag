# Phase 7 — Evaluation with RAGAS

> **This is the phase that elevates the project above average.** So far, you have checked the quality of the system "by eye": you ask a question, look at the response, and it seems good or bad. That is fine to start with, but a rigorous project needs to **measure** its quality objectively. This is where you apply the same rigor to generative AI that you applied to models in the MLOps project: instead of relying on intuition, you measure with numbers and use those numbers to improve systematically. There is also a neat detail that maintains the project's coherence: you will also run the evaluation **locally**, using your own LLM as a judge, so not even evaluating breaks the promise of privacy. Few things distinguish an exceptional junior project from an ordinary one as much as the presence of a rigorous evaluation.

**Phase objective:** measure the system's quality rigorously.  
**Duration:** first part of week 7.  
**Upon completion, you will have:** an evaluation dataset (questions + expected answers), a report with objective metrics of your RAG's quality (faithfulness, answer relevancy, context precision, and context recall), and the experience of having used those metrics to improve the system measurably.

---

## The Big Picture

Evaluation has a clear workflow that translates quality, which was previously intuitive, into numbers:

```
   Evaluation dataset (questions + expected answers)
        │
        ▼
   For each question: run RAG pipeline  ──►  response + retrieved context
        │
        ▼
   RAGAS samples (question, context, response, reference)
        │
        ▼
   RAGAS evaluates with 4 metrics  ──►  using the local LLM as a judge
        │
        ▼
   Report with metrics  ──►  iterate: change something, measure, compare
```

We will use RAGAS, the reference framework for evaluating RAG systems. Add the dependencies, along with an important warning: **RAGAS changes its API frequently** (it has had breaking changes), so it is wise to pin the version once it works, ensuring your evaluation is reproducible. The code for this phase corresponds to the API from v0.2 onwards:

```bash
uv add "ragas>=0.2" langchain-ollama
```

---

## Step 1 — Why evaluate, and why "by eye" is not enough

Before diving into the code, it is worth understanding why evaluation matters so much, as it justifies this entire phase. Your RAG system has many "dials" you can adjust: chunk sizes, how many chunks to retrieve, whether you use reranking and with which model, how the prompt is drafted, and which embedding model you use. Each of these decisions affects answer quality. The question is: when you change one, does the system get better or worse?

Judging it "by eye" is unreliable and does not scale. You test three questions, think it runs better, but what if you made things worse for ten other questions you did not test? What if the change you thought was good actually caused harm? Without a way to measure, you are tuning blindly, guided by an impression that can mislead you. Evaluation turns that **guesswork into engineering**: you establish objective metrics, measure before and after each change, and keep what is demonstrably an improvement. It is the exact same mindset you applied to models in the MLOps project, where you did not say "the model seems good," but rather measured its PR-AUC. Here you do the same with your RAG, and being able to say "my system has a faithfulness of 0.85, and I raised it from 0.72 by improving the prompt" is what proves you work with the rigor of an engineer.

---

## Step 2 — The four RAGAS metrics

RAGAS measures four main metrics, and understanding them well is key because they have a very useful property: they are divided between the **two halves** of your system (retrieval and generation), allowing you to know not only *if* something is wrong, but *where*.

Two metrics measure the quality of **retrieval**. **Context precision** measures whether the retrieved chunks are relevant and well-ordered (was the context you retrieved pertinent, or did you bring in noise?). **Context recall** measures whether you retrieved all the necessary information to answer (was a crucial chunk left out?). Together, they tell you how well your retrieval from Phase 4 is working.

The other two measure the quality of **generation**. **Faithfulness** measures whether the response is truly supported by the retrieved context, without making things up (did the model hallucinate, or did it stick to the evidence?). This is the metric that verifies your grounding from Phase 5 works. And **answer relevancy** measures whether the response truly addresses the question (did it answer what was asked, or did it go off on a tangent?).

The power of this breakdown is **diagnosis**. If your answers are poor, the metrics tell you why: if context recall is low, the problem lies in retrieval (you are not bringing in what is needed), meaning you need to improve chunking or increase the number of chunks; if faithfulness is low, the problem lies in generation (the model is making things up), meaning you need to tighten the prompt. Without this separation, you would know something is failing, but not what to fix. With it, you go straight to the cause. This is what turns evaluation into a tool for improvement rather than just a number.

---

## Step 3 — The evaluation dataset

To measure, you need an **evaluation dataset**: a collection of questions about your repository, each with its expected answer (the "reference" or *ground truth*). The reference is necessary because some metrics compare what the system answers and retrieves against what it *should* have. Here lies another advantage of having indexed your own MLOps project: since you know that code inside out, you know what the correct answers are, making it easy to build a solid evaluation dataset. Create `eval/questions.yaml`:

```yaml
- question: "¿Cómo se validan los datos de entrada en el pipeline?"
  reference: "Mediante un esquema de Pandera en el módulo de validación, que comprueba el tipo y rango de cada columna antes de procesar."

- question: "¿Qué métrica se usa para evaluar el modelo y por qué?"
  reference: "Se usa PR-AUC en lugar de ROC-AUC, porque el dataset está muy desbalanceado y PR-AUC refleja mejor el rendimiento en la clase minoritaria."

- question: "¿Cómo se detecta el drift en producción?"
  reference: "Con Evidently, comparando la distribución de los datos recientes (de los logs de predicciones) con la del entrenamiento."

# ... add between 10 and 30 representative questions
```

A good evaluation dataset covers representative questions of how the assistant will be used: some explanatory, some about specific details, some about design decisions. A practical tip: since evaluation runs the pipeline and the judge (the local LLM) for each question, and local models take time, keep it at a reasonable size (between 10 and 30 questions) so evaluation does not take forever. It is enough to obtain a useful signal.

---

## Step 4 — The local judge: evaluating without compromising privacy

Here is a crucial and highly characteristic point of this project. RAGAS calculates its metrics using an **LLM as a judge** (it asks a model, for example, "is this response supported by this context?"). By default, RAGAS uses OpenAI for this judge, which would mean sending your data to an external server, breaking the central promise of your project.

The solution, true to the "everything local" philosophy, is to configure RAGAS to use your **local LLM** (Ollama) and local embeddings as the judge. This is not only possible but is a legitimate and recognized option: evaluating with a local judge maintains alignment with a privacy-centric design and is consistent with your system. It is configured by wrapping the local models in RAGAS adapters:

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from src.config import CODE_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL


def get_local_judge():
    """Configures the local LLM and embeddings as the RAGAS judge."""
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(model=CODE_MODEL, base_url=OLLAMA_BASE_URL)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    )
    return evaluator_llm, evaluator_embeddings
```

Running the evaluation itself locally is a complete detail that demonstrates coherence: your system is private end-to-end, with no exceptions—not even for measuring its own quality. It is the kind of detail an attentive reviewer appreciates.

---

## Step 5 — The evaluation script

Now you bring everything together in `src/evaluation/ragas_eval.py`: load the questions, run the RAG pipeline for each one (capturing the response and retrieved context), build the RAGAS samples, and evaluate using the four metrics with the local judge.

```python
from pathlib import Path

import yaml
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
)

from src.evaluation.judge import get_local_judge
from src.generation.llm import get_llm
from src.generation.prompt import build_prompt
from src.retrieval.retriever import retrieve


def load_questions(path: str = "eval/questions.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_pipeline(index, nodes, question: str) -> tuple[str, list[str]]:
    """Runs the RAG and returns the response and retrieved context."""
    retrieved = retrieve(index, nodes, question)
    contexts = [n.get_content() for n in retrieved]
    answer = str(get_llm().complete(build_prompt(question, retrieved)))
    return answer, contexts


def evaluate_system(index, nodes):
    questions = load_questions()
    samples = []
    for item in questions:
        answer, contexts = run_pipeline(index, nodes, item["question"])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=contexts,
                response=answer,
                reference=item["reference"],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    evaluator_llm, evaluator_embeddings = get_local_judge()

    return evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )


if __name__ == "__main__":
    from src.config import REPO_PATH
    from src.indexing.build_index import build_index
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_repository
    from src.settings import configure_local_models

    configure_local_models()
    nodes = chunk_documents(load_repository(REPO_PATH))
    index = build_index(REPO_PATH)

    result = evaluate_system(index, nodes)
    print(result)  # means of each metric

    # Detailed report per question
    df = result.to_pandas()
    df.to_csv("eval/results.csv", index=False)
    print("\nMeans per metric:")
    print(df.mean(numeric_only=True))
```

The script loops through each question, runs the pipeline (retrieving context and generating the response), and assembles the RAGAS samples with the four elements that the metrics need: the question, the retrieved context, the generated response, and the expected reference. Then RAGAS, using the local judge, calculates the four metrics. In the end, you get the averages and a detailed report per question in `eval/results.csv`. *(Note: RAGAS frequently changes its API, so if the exact names of the metric classes differ in your version, consult its metrics module; the concept of the four metrics remains the same).*

---

## Step 6 — Iterate: Improving in a Measurable Way

Here lies the true value of evaluation and the criterion of this phase: using metrics to **improve in a measurable way**, not "by eye". Metrics are not meant to yield a one-time number, but to feed an improvement loop. The flow is: you measure, see which metric is weak, change *one* thing that should improve it, measure again, and keep the change only if the numbers confirm it.

This is where the diagnostic value of Step 2 becomes gold, because each weak metric points to a specific fix. If **context recall** is low, you are not retrieving everything necessary: try increasing the number of retrieved chunks or reviewing chunking. If **context precision** is low, you are bringing in noise: reranking should help, or adjusting the hybrid search. If **faithfulness** is low, the model is making things up: tighten the grounding instructions in the prompt. If **answer relevancy** is low, the model goes off on tangents: fine-tune the prompt to make it more direct.

For example, you might measure and see a faithfulness of 0.72; you suspect that the model sometimes drifts from the context, so you reinforce the instruction to stick to the chunks in the prompt; you measure again and get 0.85. Now you *know*, with a number, that the change improved the system, rather than just intuiting it. This cycle (measure, change, measure again) is the essence of AI system engineering, and documenting it in your README ("improved faithfulness from 0.72 to 0.85 by tightening grounding") is exactly what demonstrates rigor.

---

## Step 7 — The report and its nuances

The deliverable for this phase is an **evaluation report** with the metrics, which will live in your repository (`eval/results.csv`) and which you will highlight in the Phase 8 README as objective evidence of your system's quality. This is a massive differentiator: most junior RAG projects lack any evaluation, so showing concrete metrics puts you far ahead.

That being said, it is wise to present metrics honestly with respect to their nuances, as acknowledging limitations is, in itself, a sign of rigor. RAGAS scores are a useful reference, but not an absolute truth: they rely on an LLM as a judge, which is not perfect or calibrated against human criteria, and small evaluation datasets introduce noise. For this reason, metrics are used as a **guide** for improvement and as comparative evidence (before/after a change), not as a definitive, unquestionable grade. Showing that you understand this (that you use them with judgment, knowing what they measure and what they do not) demonstrates a maturity that goes far beyond knowing how to run the tool.

---

## Verification: The "Definition of Done"

The phase is complete when the following are met:

- [ ] An evaluation dataset exists with questions and expected answers about the repository.
- [ ] `ragas_eval.py` evaluates the system using the four RAGAS metrics.
- [ ] The evaluation uses the local LLM as a judge, maintaining privacy.
- [ ] A report is generated with the metrics (averages and detail per question).
- [ ] You have used the metrics to improve the system and have verified the improvement with numbers.
- [ ] **The key test:** You have objective metrics for your system and have used the evaluation to improve it measurably, not "by eye".

The key test is the measurable improvement cycle: if you have objective numbers of your RAG's quality and can show that a specific change improved a specific metric, you have stopped tuning blindly and started doing real engineering. That ability to measure and improve with rigor is exactly what elevates your project, and shows that you apply the same production-level seriousness to generative AI as you would to any other system.

---

## Deliverables and What Comes Next

By closing Phase 7, you have added what distinguishes your project most from the average: a rigorous evaluation. You have objective metrics for the quality of your RAG, split between retrieval and generation so they tell you not only if something goes wrong, but where. You have also used those metrics to improve the system measurably. All of this, moreover, is evaluated locally, staying true to privacy. You have demonstrated that you work with the same engineering rigor as in the MLOps project, now applied to generative AI.

The next and final step, **Phase 8**, showcases the project's value: **packaging and presentation**. You will containerize the system using Docker (alongside Ollama), write the final README (with this phase's RAGAS metrics as proof of quality), record a video showing the assistant in action, and write a post about the project. All the technical work across the eight phases now needs a display window. You have gone from "I can prove with numbers how good my system is" to being on the verge of "I am presenting it to leave companies thoroughly impressed."
