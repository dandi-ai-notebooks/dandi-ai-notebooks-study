#!/bin/bash

set -ex

DANDISET_ID=001349
DANDISET_VERSION=draft
CHAT_ID=5afc0883f430b14ee8a901c92a7d295c0b65e1dd
CHAT_ID_2=${CHAT_ID:0:2}
CHAT_ID_8=${CHAT_ID:0:8}
# MODEL="anthropic/claude-3.7-sonnet"
# MODEL="openai/gpt-4.1-mini"
MODEL="openai/gpt-4.1"
MODEL_SECOND_PART=${MODEL#*/}
PROMPT="h-1"

CHAT_URL="https://neurosift.org/dandiset-explorer-chats/$DANDISET_ID/$DANDISET_VERSION/chats/$CHAT_ID_2/$CHAT_ID/chat.json"
PROMPT_PATH=prompts/prompt-$PROMPT.txt
OUTPUT="notebooks/dandisets/$DANDISET_ID/${DANDISET_VERSION}/$CHAT_ID_8/$MODEL_SECOND_PART/$PROMPT"

python scripts/create_notebook.py --dandiset $DANDISET_ID --version $DANDISET_VERSION --model $MODEL --chat $CHAT_URL --prompt $PROMPT_PATH --output $OUTPUT