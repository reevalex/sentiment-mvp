#!/bin/zsh

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was amazing!"}'