#!/usr/bin/env python3
"""Script to list available Bedrock models"""

from bedrock_client import bedrock_client

models = bedrock_client.list_available_models()
print("Available Bedrock models:")
for model in sorted(models):
    print(f"  - {model}")
