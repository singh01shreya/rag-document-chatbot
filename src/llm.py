"""Provider wrapper.

Every other module gets its chat model and embeddings through the two
functions below instead of importing LangChain's OpenAI classes directly.
To swap providers (e.g. to a local Ollama model so the app runs without an
API key), change the bodies of these two functions only — nothing else in
the codebase needs to know which provider is in use.
"""

import os

from langchain_ollama import ChatOllama, OllamaEmbeddings

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")


def get_chat_model(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, temperature=temperature)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)