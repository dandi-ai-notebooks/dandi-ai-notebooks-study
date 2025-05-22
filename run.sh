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
    "001349 0.250520.1729 90f7cc7c8af637c7bbef0bb8d031d904c6367e79"
    "001354 0.250312.0036 4fba179c5e91d523298cbec78993c2448eb15e1e"
    "001433 0.250507.2356 daf6f524d0a0c520ac0c3e856d5162ae0dcf7dd2"
    "000563 0.250311.2145 75ecdb7baaa6e5334b161865b3df90150c2a1d3f"
    "001361 0.250406.0045 5c7338bfd1c3632f146b7fa004d39328a790a4a3"
    "001366 0.250324.1603 cbe62122b7755e61a489a61c72ede7dbcfd63b48"
    "001359 0.250401.1603 28668318a5283680bb2c5e48db13ae2b4031c10a"
    "001375 0.250406.1855 5fe912e30a4a72bcc811998379756261432889bc"
    "001174 0.250331.2218 8b5d2333dbafc85c105768f2773ed87b01466150"
    "000690 0.250326.0015 d8bfae1f2edb4521deb0445376279f1aabe65355"
    "001195 0.250408.1733 ae31fe05a7d09bd0797b31a0d7cae5a835a0959d"
    "000617 0.250311.1615 ca0caac9e3959f2601d73837a00ffb76968f780a"
)

# Process each dandiset
for config in "${DANDISETS[@]}"; do
    read -r id version chat_id <<< "$config"
    process_dandiset "$id" "$version" "$chat_id"
done
