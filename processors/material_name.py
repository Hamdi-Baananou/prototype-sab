import re
import requests
import yaml
import streamlit as st
from pathlib import Path
from typing import Tuple

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def calculate_confidence(reasoning: str) -> int:
    """Calculate confidence score based on completed steps"""
    checkmarks = reasoning.count('✓')
    total_steps = 5  # Total steps in the reasoning chain
    return min(100, int((checkmarks / total_steps) * 100))

def parse_response(response_text: str) -> Tuple[str, int]:
    """Extract material name and confidence from API response"""
    try:
        # Split reasoning and material name
        reasoning_section = response_text.split("REASONING:")[1].split("MATERIAL NAME:")[0].strip()
        material_section = response_text.split("MATERIAL NAME:")[1].strip()
        
        # Extract material name
        material = "NOT FOUND"
        if material_section:
            material = re.search(r'^([A-Z0-9]+|NOT FOUND)', material_section).group(0)
        
        # Calculate confidence
        confidence = calculate_confidence(reasoning_section)
        
        return material, confidence
        
    except Exception as e:
        st.error(f"Response parsing failed: {str(e)}")
        return "NOT FOUND", 0

def analyze(text: str) -> Tuple[str, int]:
    """Extract material name with confidence score"""
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)['attributes']['material_name']
            
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": config['system_prompt']
                },
                {
                    "role": "user",
                    "content": f"COMBINED DOCUMENTS:\n{text[:30000]}"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 512,
            "top_p": 0.9
        }

        response = requests.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        raw_result = response.json()['choices'][0]['message']['content'].strip()
        st.write(f"Raw API Response:\n{raw_result}")  # For debugging
        
        return parse_response(raw_result)
        
    except Exception as e:
        st.error(f"Material analysis failed: {str(e)}")
        return "ANALYSIS_ERROR", 0

__all__ = ['analyze']