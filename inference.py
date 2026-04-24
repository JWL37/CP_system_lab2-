from __future__ import annotations

from typing import Sequence

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def query_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Не удалось подключиться к API Ollama. Убедитесь, что запущен `ollama serve`."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama API вернул HTTP {response.status_code}: {response.text}"
        )

    data = response.json()
    model_output = data.get("response")
    if not isinstance(model_output, str):
        raise RuntimeError("В ответе Ollama API отсутствует строковое поле `response`.")

    return model_output.strip()


def create_report(prompts: Sequence[str], responses: Sequence[str], filename: str) -> None:
    if len(prompts) != len(responses):
        raise ValueError("Количество запросов и ответов должно совпадать.")

    lines = [
        "# Отчет по инференсу",
        "",
        f"Модель: `{MODEL_NAME}`",
        "",
        "| Запрос | Ответ LLM |",
        "|---|---|",
    ]

    for prompt, answer in zip(prompts, responses):
        safe_prompt = prompt.replace("|", "\\|").replace("\n", "<br>")
        safe_answer = answer.replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {safe_prompt} | {safe_answer} |")

    with open(filename, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def main() -> None:
    """Запустить 10 заранее заданных запросов и сохранить `report.md`."""
    prompts = [
        "Что такое машинное обучение в одном простом предложении?",
        "Кратко объясни разницу между TCP и UDP.",
        "Напиши короткое поздравление с днем рождения на русском языке.",
        "Дай 3 практических совета, как улучшить концентрацию во время учебы.",
        "Сгенерируй идею стартапа в сфере экологии в 2-3 предложениях.",
        "Напиши функцию на Python, которая проверяет, является ли число простым.",
        "Объясни, что такое REST API, простыми словами.",
        "Сочини короткое четверостишие о весне.",
        "Назови 5 способов снизить энергопотребление дома.",
        "Кратко объясни, почему контроль версий важен для командных проектов.",
    ]

    responses = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/10] Отправка запроса...")
        answer = query_ollama(prompt)
        responses.append(answer)

    create_report(prompts, responses, filename="report.md")
    print("Готово. Отчет сохранен в report.md")


if __name__ == "__main__":
    main()