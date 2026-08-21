"""
Helper to upload training_data.jsonl and start an OpenAI fine-tune job.

Requires OPENAI_API_KEY in your environment and the openai package installed.

Usage:
    pip install openai
    set OPENAI_API_KEY=your_key  (Windows) or export OPENAI_API_KEY=your_key
    python openai_finetune.py training_data.jsonl

This script uses the OpenAI API to upload a file and create a fine-tune job.
Note: OpenAI fine-tune APIs evolve; adapt model id and API calls to the current provider docs.
"""

import os
import sys


def main():
    try:
        import openai
    except Exception:
        print("openai package not installed. pip install openai")
        raise

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set in environment")
        return
    openai.api_key = api_key
    if len(sys.argv) < 2:
        print("Usage: python openai_finetune.py training_data.jsonl")
        return
    infile = sys.argv[1]
    print("Uploading", infile)
    # Upload file
    with open(infile, "rb") as f:
        resp = openai.File.create(file=f, purpose="fine-tune")
    print("Upload response:", resp)
    file_id = resp["id"]
    # Create fine-tune
    # NOTE: replace model with desired base model
    try:
        ft = openai.FineTune.create(training_file=file_id, model="gpt-3.5-turbo")
        print("Fine-tune started:", ft)
    except Exception as e:
        print("Fine-tune create failed:", e)


if __name__ == "__main__":
    main()
