"""Groq configuration and isolated per-article summarization."""
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.prompts import load_prompt

SUMMARY_PROMPT = load_prompt("summarizer")


def create_llm():
    """Create Groq after .env is loaded; bound timeout, retries and output."""
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise ValueError("Set GROQ_API_KEY in .env")
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                    temperature=0, timeout=60, max_retries=2, max_tokens=2048)


async def summarize_article(llm, article):
    """Send only this article in a fresh pair of messages."""
    response = await llm.ainvoke([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=f"Заголовок: {article['title']}\n\nСтатья:\n{article['text']}")])
    if response.response_metadata.get("finish_reason") == "length":
        raise ValueError("Summary truncated; increase max_tokens")
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("Empty summary from Groq")
    return response.content.strip()
