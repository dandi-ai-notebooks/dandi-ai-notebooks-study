#!/bin/bash

set -ex

DANDISET_ID=001349
DANDISET_VERSION=draft
CHAT_ID=5afc0883f430b14ee8a901c92a7d295c0b65e1dd
# MODEL="anthropic/claude-3.7-sonnet"
# MODEL="openai/gpt-4.1-mini"
MODEL="openai/gpt-4.1"
PROMPT="h-1"

CHAT_ID_2=${CHAT_ID:0:2}
CHAT_ID_8=${CHAT_ID:0:8}
CHAT_URL="https://neurosift.org/dandiset-explorer-chats/$DANDISET_ID/$DANDISET_VERSION/chats/$CHAT_ID_2/$CHAT_ID/chat.json"
MODEL_SECOND_PART=${MODEL#*/}
PROMPT_PATH=prompts/prompt-$PROMPT.txt
OUTPUT="notebooks/dandisets/$DANDISET_ID/${DANDISET_VERSION}/$CHAT_ID_8/$MODEL_SECOND_PART/$PROMPT"
python scripts/create_notebook.py --dandiset $DANDISET_ID --version $DANDISET_VERSION --model $MODEL --chat $CHAT_URL --prompt $PROMPT_PATH --output $OUTPUT

DANDISET_ID=001354
DANDISET_VERSION=draft
CHAT_ID=72e0ecd3aa4ccf51f2775a40172cb879089dc222
MODEL="openai/gpt-4.1"
PROMPT="h-1"

CHAT_ID_2=${CHAT_ID:0:2}
CHAT_ID_8=${CHAT_ID:0:8}
CHAT_URL="https://neurosift.org/dandiset-explorer-chats/$DANDISET_ID/$DANDISET_VERSION/chats/$CHAT_ID_2/$CHAT_ID/chat.json"
MODEL_SECOND_PART=${MODEL#*/}
PROMPT_PATH=prompts/prompt-$PROMPT.txt
OUTPUT="notebooks/dandisets/$DANDISET_ID/${DANDISET_VERSION}/$CHAT_ID_8/$MODEL_SECOND_PART/$PROMPT"
python scripts/create_notebook.py --dandiset $DANDISET_ID --version $DANDISET_VERSION --model $MODEL --chat $CHAT_URL --prompt $PROMPT_PATH --output $OUTPUT

DANDISET_ID=001433
DANDISET_VERSION=draft
CHAT_ID=0ef8339a2b5beef133a3e953e6de3da3ca554b1d
MODEL="openai/gpt-4.1"
PROMPT="h-1"

CHAT_ID_2=${CHAT_ID:0:2}
CHAT_ID_8=${CHAT_ID:0:8}
CHAT_URL="https://neurosift.org/dandiset-explorer-chats/$DANDISET_ID/$DANDISET_VERSION/chats/$CHAT_ID_2/$CHAT_ID/chat.json"
MODEL_SECOND_PART=${MODEL#*/}
PROMPT_PATH=prompts/prompt-$PROMPT.txt
OUTPUT="notebooks/dandisets/$DANDISET_ID/${DANDISET_VERSION}/$CHAT_ID_8/$MODEL_SECOND_PART/$PROMPT"
python scripts/create_notebook.py --dandiset $DANDISET_ID --version $DANDISET_VERSION --model $MODEL --chat $CHAT_URL --prompt $PROMPT_PATH --output $OUTPUT

DANDISET_ID=000563
DANDISET_VERSION=0.250311.2145
CHAT_ID=fff024797bef5d45054916026a539bdcd19c8771
MODEL="openai/gpt-4.1"
PROMPT="h-1"

CHAT_ID_2=${CHAT_ID:0:2}
CHAT_ID_8=${CHAT_ID:0:8}
CHAT_URL="https://neurosift.org/dandiset-explorer-chats/$DANDISET_ID/$DANDISET_VERSION/chats/$CHAT_ID_2/$CHAT_ID/chat.json"
MODEL_SECOND_PART=${MODEL#*/}
PROMPT_PATH=prompts/prompt-$PROMPT.txt
OUTPUT="notebooks/dandisets/$DANDISET_ID/${DANDISET_VERSION}/$CHAT_ID_8/$MODEL_SECOND_PART/$PROMPT"
python scripts/create_notebook.py --dandiset $DANDISET_ID --version $DANDISET_VERSION --model $MODEL --chat $CHAT_URL --prompt $PROMPT_PATH --output $OUTPUT

# https://dandi-ai-notebooks.github.io/dandiset-explorer/chat?dandisetId=001361&dandisetVersion=0.250406.0045&chatId=09466fb7ab663528a1893423a1804e7425ff6178
DANDISET_ID=001361
DANDISET_VERSION=0.250406.0045
CHAT_ID=09466fb7ab663528a1893423a1804e7425ff6178
MODEL="openai/gpt-4.1"
PROMPT="h-1"

CHAT_ID_2=${CHAT_ID:0:2}
CHAT_ID_8=${CHAT_ID:0:8}
CHAT_URL="https://neurosift.org/dandiset-explorer-chats/$DANDISET_ID/$DANDISET_VERSION/chats/$CHAT_ID_2/$CHAT_ID/chat.json"
MODEL_SECOND_PART=${MODEL#*/}
PROMPT_PATH=prompts/prompt-$PROMPT.txt
OUTPUT="notebooks/dandisets/$DANDISET_ID/${DANDISET_VERSION}/$CHAT_ID_8/$MODEL_SECOND_PART/$PROMPT"
python scripts/create_notebook.py --dandiset $DANDISET_ID --version $DANDISET_VERSION --model $MODEL --chat $CHAT_URL --prompt $PROMPT_PATH --output $OUTPUT