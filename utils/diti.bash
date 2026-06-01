#!/bin/bash

URL="http://localhost:6969/message"

echo "-----------------------------------"

while true; do
    read -p "You: " input

    [[ "$input" == "exit" ]] && echo "Bye." && break
    [[ -z "$input" ]] && continue

    response=$(curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d "{\"text\": $(echo "$input" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}")

    reply=$(echo "$response" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["response"])')

    echo "Diti: $reply"
    echo ""
done
