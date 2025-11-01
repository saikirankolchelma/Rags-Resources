🧠 What is RAG?

  RAG (Retrieval-Augmented Generation) is a framework that combines information retrieval and text generation to make Large Language Models (LLMs) more accurate, up-to-date, and explainable.

In simple terms:

Instead of expecting the LLM to “know everything,” we retrieve relevant external knowledge and augment the LLM’s input with it before generating the final answer.

⚙️ The Core Idea

Traditional LLMs (like GPT, Claude, etc.) are trained on fixed data — they can’t know anything after their last training cutoff.

So, if you ask:

“Summarize yesterday’s Google earnings call.”

→ The base model can’t answer, because that info wasn’t part of its training data.

RAG fixes this by doing two main things:

Retrieve: Fetch the most relevant documents/passages from a knowledge source (database, PDF, website, etc.)

Generate: Pass both the user’s question and the retrieved context into the LLM, so it can generate an informed, grounded answer.
