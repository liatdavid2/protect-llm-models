# Protect LLM Models

A lightweight security gateway for language models.

The system protects an RAG/LLM-style API with multiple guards before and after model inference:

- Prompt Injection Input Guard
- Harmful Content Input Guard
- PII Output Guard
- System Prompt Leakage Output Guard

The API exposes a single `/chat` endpoint and measures latency for every stage.

## What this project does

This project adds security layers around a model served through Ollama.

Flow:

1. Check the user prompt with the Prompt Injection guard
2. Check the user prompt with the Harmful Content guard
3. Run the model
4. Check the generated output with the PII Output guard
5. Check the generated output with the System Prompt Leakage Output guard
6. Return the response only if all enabled stages pass

Each stage can be disabled dynamically through `disabled_steps`, and latency is reported for every step.

## Architecture

This architecture is practical for secured inference because it separates risks by stage:

- **Input guards** stop unsafe or adversarial prompts before the model runs
- **Output guards** inspect the generated text before it is returned
- **Per-step latency** makes it easy to understand performance overhead
- **Disabled steps** make testing and ablation simple
- **Independent guard modules** make retraining and replacement easy

Main idea:

`User Prompt -> Input Guards -> Model -> Output Guards -> Final Response`

## Guards

### 1. Prompt Injection Input Guard
Detects malicious prompts such as:
- instruction override attempts
- attempts to ignore previous rules
- attempts to extract hidden instructions
- attempts to bypass policy or reveal internal behavior

### 2. Harmful Content Input Guard
Detects unsafe or malicious user requests such as:
- violent wrongdoing
- illegal harmful instructions
- clearly dangerous requests

### 3. PII Output Guard
Checks generated model output for personal or sensitive information leakage.

This guard uses two complementary ideas:

#### Regex-based detection
Useful for structured patterns such as:
- email addresses
- phone numbers
- ID-like values
- credit-card-like patterns

Regex is fast and precise for known formats.

#### Model-based detection
Useful for contextual or flexible cases such as:
- partial personal details
- natural language descriptions of private information
- cases that do not follow a strict format

Using both regex and a trained model provides better coverage than using only one of them.

### 4. System Prompt Leakage Output Guard
Checks whether the model output contains leaked internal instructions, hidden policies, system prompt fragments, or other protected internal content.

This is the final output protection layer.

## Datasets

Each guard is trained on a real dataset, not synthetic placeholder data.

Training commands:

```bash
python harmful_content_input_guard\train.py
python prompt_injection_input_guard\train.py
python pii_output_guard\train.py
python system_prompt_leakage_output_guard\train.py