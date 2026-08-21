"""
Example showing how to integrate TOON into a LangChain pipeline.

This example is resilient to missing langchain: if langchain is installed, it shows a
minimal Chain usage; otherwise it demonstrates how to build a prompt and call your LLM
client directly.

To actually run with LangChain + OpenAI:
  pip install langchain openai
  set OPENAI_API_KEY=... (Windows: setx OPENAI_API_KEY "your_key")

Then run: python langchain_integration.py
"""

from __future__ import annotations

import os

from sdk.toon_extended import build_prompt_for_llm

sample = [
    {"id": 1, "name": "Widget A", "price": 9.99},
    {"id": 2, "name": "Widget B", "price": 15.5},
]

instruction = "Return ids and names for items with price > 10 as JSON array of objects {id,name}."

prepared = build_prompt_for_llm(instruction, sample, include_schema=True)
print("Prepared prompt (short):")
print(prepared["prompt_text"][:1000])
print("\nToken estimate:", prepared["prompt_tokens_estimate"])

try:
    # Attempt LangChain usage if available
    from langchain import LLMChain, PromptTemplate
    from langchain.chat_models import ChatOpenAI

    template = "{prompt}"
    prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
    # Create a ChatOpenAI instance (requires OPENAI_API_KEY)
    model = ChatOpenAI(model_name=os.environ.get("TOON_DEFAULT_MODEL", "gpt-4"), temperature=0)
    chain = LLMChain(llm=model, prompt=prompt_template)
    resp = chain.run(prepared["prompt_text"])
    print("\nLangChain response:")
    print(resp)
except Exception as e:
    print("\nLangChain not available or failed to run. Provide your own client. Error:")
    print(str(e))
    print("\nExample using OpenAI directly:")
    print(
        "Use build_prompt_for_llm() to create the prompt text and send it as a single "
        "message to your model client (OpenAI, Anthropic, etc.)"
    )
