#!/bin/bash

set -ex

MODEL="openai/gpt-4.1"
PROMPT="h-1"
PROMPT_PATH="prompts/prompt-$PROMPT.txt"
MODEL_SECOND_PART=${MODEL#*/}

process_dandiset() {
    local DANDISET_ID=$1
    local DANDISET_VERSION=$2
    local CHAT_ID=$3

    local CHAT_ID_2=${CHAT_ID:0:2}
    local CHAT_ID_8=${CHAT_ID:0:8}
    local CHAT_URL="https://neurosift.org/dandiset-explorer-chats/$DANDISET_ID/$DANDISET_VERSION/chats/$CHAT_ID_2/$CHAT_ID/chat.json"
    local OUTPUT="notebooks/dandisets/$DANDISET_ID/${DANDISET_VERSION}/$CHAT_ID_8/$MODEL_SECOND_PART/$PROMPT"

    python scripts/create_notebook.py \
        --dandiset "$DANDISET_ID" \
        --version "$DANDISET_VERSION" \
        --model "$MODEL" \
        --chat "$CHAT_URL" \
        --prompt "$PROMPT_PATH" \
        --output "$OUTPUT"
}

# Array of dandiset configurations: [id version chat_id]
declare -a DANDISETS=(
    "001349 draft 5afc0883f430b14ee8a901c92a7d295c0b65e1dd"
    "001354 draft 72e0ecd3aa4ccf51f2775a40172cb879089dc222"
    "001433 draft 0ef8339a2b5beef133a3e953e6de3da3ca554b1d"
    "000563 0.250311.2145 fff024797bef5d45054916026a539bdcd19c8771"
    "001361 0.250406.0045 09466fb7ab663528a1893423a1804e7425ff6178"
    "001366 0.250324.1603 6ee9f00b7aad91348568cbe0c5dae2cbcb21de2e"
    "001359 0.250401.1603 28668318a5283680bb2c5e48db13ae2b4031c10a"
    "001375 0.250406.1855 5fe912e30a4a72bcc811998379756261432889bc"
    "001174 0.250331.2218 8b5d2333dbafc85c105768f2773ed87b01466150"
    "000690 0.250326.0015 d8bfae1f2edb4521deb0445376279f1aabe65355"
)

# Process each dandiset
for config in "${DANDISETS[@]}"; do
    read -r id version chat_id <<< "$config"
    process_dandiset "$id" "$version" "$chat_id"
done
