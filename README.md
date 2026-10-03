# Notes Chatbot (RAG)

A small chatbot that answers questions from your own `.txt`/`.md` notes using an LLM.

## How it works
1. `rag.py` splits your notes into overlapping chunks and indexes them with TF-IDF.
2. For each question it retrieves the top-k chunks.
3. `app.py` sends those chunks plus the question to an LLM (Anthropic API), which must answer
   only from the context and cite passages like [1], or say it doesn't know.
4. `eval.py` checks whether retrieval finds the right file for a set of test questions.

## Run it
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python eval.py            # retrieval check, no API key needed
streamlit run app.py
```
Put your own notes in `notes/` (replace the samples) and update `eval_set.json`.

## Ideas to extend (make it yours)
- Replace TF-IDF in `rag.py` with embeddings and compare hit@3 using `eval.py`
- Add PDF support, or a "no relevant context" threshold
- Add answer-quality evals (e.g. check that cited sources support the answer)
- Log latency and token cost per question
