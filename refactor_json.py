import json
import re
import argparse


def clean_text(text):
    """
    Cleans the text by removing extra spaces, line breaks, and correcting special characters.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespaces
    text = re.sub(r"[\r\n]+", " ", text)  # Remove line breaks
    # Fix common encoding issues and special characters
    text = (
        text.replace("Ã©", "é")
        .replace("Ã¡", "á")
        .replace("Ã§", "ç")
        .replace("Ã£", "ã")
        .replace("Ãµ", "õ")
        .replace("Ãª", "ê")
        .replace("Ãí", "í")
        .replace("Ã", "í")
        .replace("â", "â")
        .replace("€", "€")
        .replace("™", "™")
        .replace("Â", "")  # Remove common garbage character
        .replace("&#39;", "'")  # Fix HTML entities
        .replace("\xa0", " ")  # Remove non-breaking space
        .replace("â€", "")  # Remove another garbage character
    )
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Improve clarity and naturalness
    text = text.replace("O que é", "Como funciona")
    text = text.replace("Qual é a diferença entre", "Diferença entre")
    text = text.replace("Quais são os benefícios de", "Benefícios de")

    # Expand abbreviations (a placeholder, expand more as needed)
    text = text.replace("ex:", "por exemplo:")

    return text


def rewrite_text(text):
    """
    Rewrites the text to improve clarity and quality.
    This is a placeholder for more sophisticated rewriting logic.
    """
    # Remove repetitive phrases
    text = re.sub(r"(.*)\1+", r"\1", text)

    # Make the text more concise
    # (This is a placeholder for more sophisticated conciseness logic)
    text = text.replace("em relação a", "sobre")
    text = text.replace("no que se refere a", "sobre")

    # Correct grammar
    # (This is a placeholder for more sophisticated grammar correction logic)
    if text.startswith("A "):
        text = text.replace("A ", "O ")  # Correct common gender agreement issue

    return text


def is_valid_content(text):
    """
    Checks if the content is valid (not vague, incomplete, or repetitive).
    """
    text = text.strip()
    if not text:
        return False
    if len(text) < 15:  # Minimum length check
        return False
    if text.lower() in [
        "sim",
        "não",
        "ok",
        "obrigado",
        "obrigada",
    ]:  # Remove vague answers
        return False
    # Check for repetitive content
    words = text.lower().split()
    if len(words) > 5:
        from collections import Counter

        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        if (
            most_common_count / len(words) > 0.5
        ):  # If a word appears more than 50% of the time
            return False
    return True


def process_data(input_file, output_file):
    """
    Processes the input JSON file, corrects the objects, and generates a new JSON file.
    """
    total_read = 0
    total_discarded = 0
    total_corrected = 0

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_read = len(data)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{input_file}' is not a valid JSON.")
        return
    except Exception as e:
        print(f"Unexpected error when reading the file: {e}")
        return

    processed_objects = []

    for obj in data:
        # Check if the required keys exist
        if not all(
            key in obj
            for key in [
                "id",
                "categoria",
                "pergunta_principal",
                "perguntas_relacionadas",
                "resposta",
                "palavras_chave",
            ]
        ):
            total_discarded += 1
            continue

        corrected = False
        cleaned_obj = {}

        # Rewrite and clean the 'pergunta_principal' field
        if "pergunta_principal" in obj:
            original_pergunta_principal = obj["pergunta_principal"]
            pergunta_principal = rewrite_text(clean_text(obj["pergunta_principal"]))
            if is_valid_content(pergunta_principal):
                cleaned_obj["pergunta_principal"] = pergunta_principal
                if cleaned_obj["pergunta_principal"] != original_pergunta_principal:
                    corrected = True
            else:
                total_discarded += 1
                continue  # Discard if not valid

        # Rewrite and clean the 'resposta' field
        if "resposta" in obj:
            original_resposta = obj["resposta"]
            resposta = rewrite_text(clean_text(obj["resposta"]))
            if is_valid_content(resposta):
                cleaned_obj["resposta"] = resposta
                if cleaned_obj["resposta"] != original_resposta:
                    corrected = True
            else:
                total_discarded += 1
                continue  # Discard if not valid

        # Rewrite and clean the 'perguntas_relacionadas' field
        if "perguntas_relacionadas" in obj and isinstance(
            obj["perguntas_relacionadas"], list
        ):
            original_perguntas_relacionadas = obj["perguntas_relacionadas"]
            perguntas_relacionadas = [
                rewrite_text(clean_text(item)) for item in obj["perguntas_relacionadas"]
            ]
            valid_perguntas_relacionadas = [
                item for item in perguntas_relacionadas if is_valid_content(item)
            ]
            cleaned_obj["perguntas_relacionadas"] = valid_perguntas_relacionadas
            if cleaned_obj["perguntas_relacionadas"] != original_perguntas_relacionadas:
                corrected = True

        for key, value in obj.items():
            # Remove extra whitespaces from strings
            if isinstance(value, str):
                original_value = value
                value = clean_text(value)
                if value != original_value:
                    corrected = True
            # Remove empty lists
            if isinstance(value, list) and not value:
                continue  # Remove empty fields
            if key not in cleaned_obj:
                cleaned_obj[key] = value

        # Merge or remove duplicate entries in "perguntas_relacionadas" and "palavras_chave" fields
        if "perguntas_relacionadas" in cleaned_obj and isinstance(
            cleaned_obj["perguntas_relacionadas"], list
        ):
            cleaned_obj["perguntas_relacionadas"] = list(
                dict.fromkeys(cleaned_obj["perguntas_relacionadas"])
            )  # Remove duplicates

        if "palavras_chave" in cleaned_obj and isinstance(
            cleaned_obj["palavras_chave"], list
        ):
            cleaned_obj["palavras_chave"] = list(
                dict.fromkeys(cleaned_obj["palavras_chave"])
            )  # Remove duplicates

        # Add a new field "embedding_input"
        if "pergunta_principal" in cleaned_obj and "resposta" in cleaned_obj:
            related_questions = ", ".join(cleaned_obj.get("perguntas_relacionadas", []))
            cleaned_obj["embedding_input"] = (
                f"passage: {cleaned_obj['pergunta_principal']} {cleaned_obj['resposta']} {related_questions}"
            )
            corrected = True

        if corrected:
            total_corrected += 1

        processed_objects.append(cleaned_obj)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_objects, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to file: {e}")
        return

    print(f"Total objects read: {total_read}")
    print(f"Total objects discarded: {total_discarded}")
    print(f"Total objects corrected: {total_corrected}")


if __name__ == "__main__":
    input_file = r"S:/Code/LangGraph_study/HandsOn/data/faq.json"
    output_file = r"S:/Code/LangGraph_study/HandsOn/data/faq_corrected.json"

    process_data(input_file, output_file)
