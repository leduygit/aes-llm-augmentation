#!/usr/bin/env python3
"""Generate flat CSV synthetic IELTS Task 2 essays with OpenAI or Claude."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = PROJECT_DIR / "synthetic_assets"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "synthetic"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-0"


class TextProvider(Protocol):
    name: str
    model: str

    def complete(self, system_content: str, user_content: str, max_tokens: int, temperature: float) -> str:
        ...


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def parse_bands(value: str) -> List[int]:
    try:
        bands = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid band specification: {value}") from exc
    invalid = [band for band in bands if band not in range(5, 10)]
    if invalid:
        raise argparse.ArgumentTypeError(f"Bands must be integers from 5 to 9; got {invalid}")
    return bands


def retry_call(func: Callable, max_retries: int, delay: float, *args, **kwargs) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries:
                print(f"Failed after {max_retries} attempts: {exc}")
                raise
            print(f"Attempt {attempt} failed: {exc}. Retrying in {delay} seconds...")
            time.sleep(delay)


def load_dotenv_or_exit() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Error: python-dotenv is required to load .env. Install requirements.txt or rerun setup.sh.")
        sys.exit(1)
    load_dotenv(PROJECT_DIR / ".env")


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Set OPENAI_API_KEY in the environment or in .env")
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError:
            print("Error: The openai package is required for OpenAI generation. Install requirements.txt or rerun setup.sh.")
            sys.exit(1)
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self.client = OpenAI(api_key=api_key)

    def complete(self, system_content: str, user_content: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: Optional[str] = None):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: Anthropic API key required. Set ANTHROPIC_API_KEY in the environment or in .env")
            sys.exit(1)
        try:
            import anthropic
        except ImportError:
            print("Error: The anthropic package is required for Claude generation. Install requirements.txt or rerun setup.sh.")
            sys.exit(1)
        self.model = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system_content: str, user_content: str, max_tokens: int, temperature: float) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=system_content,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if hasattr(response, "content"):
            if isinstance(response.content, list):
                return "".join(block.text for block in response.content if hasattr(block, "text")).strip()
            return str(response.content).strip()
        return str(response).strip()


def build_provider(provider_name: str, model: Optional[str]) -> TextProvider:
    if provider_name == "openai":
        return OpenAIProvider(model=model)
    if provider_name == "claude":
        return ClaudeProvider(model=model)
    raise ValueError(f"Unsupported provider: {provider_name}")


class SyntheticIELTSGenerator:
    def __init__(
        self,
        asset_dir: Path,
        provider: TextProvider,
        temperature: float,
        max_retries: int,
        retry_delay: float,
    ):
        self.asset_dir = asset_dir
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._band_descriptions: Optional[Dict[str, Dict[str, str]]] = None
        self._fewshot_df: Optional[pd.DataFrame] = None
        self._questions_df: Optional[pd.DataFrame] = None

    def require_assets(self) -> None:
        required = ["fewshot-data.csv", "question.csv", "band_descriptions.json"]
        missing = [name for name in required if not (self.asset_dir / name).exists()]
        if missing:
            missing_text = ", ".join(str(self.asset_dir / name) for name in missing)
            raise FileNotFoundError(f"Missing required synthetic generation assets: {missing_text}")

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._fewshot_df is None or self._questions_df is None:
            self.require_assets()
            self._fewshot_df = pd.read_csv(self.asset_dir / "fewshot-data.csv")
            self._questions_df = pd.read_csv(self.asset_dir / "question.csv")
        return self._fewshot_df, self._questions_df

    def load_band_descriptions(self) -> Dict[str, Dict[str, str]]:
        if self._band_descriptions is None:
            self.require_assets()
            with (self.asset_dir / "band_descriptions.json").open("r", encoding="utf-8") as f:
                self._band_descriptions = json.load(f)
        return self._band_descriptions

    def complete_with_retry(self, system_content: str, user_content: str, max_tokens: int) -> str:
        return retry_call(
            self.provider.complete,
            self.max_retries,
            self.retry_delay,
            system_content=system_content,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )

    def get_examples_for_band(self, band: int, num_examples: int = 5) -> List[Dict[str, Any]]:
        fewshot_df, _ = self.load_data()
        band_examples = fewshot_df[fewshot_df["Overall"] == band]
        if len(band_examples) < num_examples:
            print(f"Warning: Only {len(band_examples)} examples available for band {band}")
            num_examples = len(band_examples)
        if num_examples == 0:
            raise ValueError(f"No examples found for band {band}")
        examples = []
        for _, row in band_examples.sample(n=num_examples).iterrows():
            examples.append({
                "question": row["Question"],
                "essay": row["Essay"],
                "overall": row["Overall"],
                "ta": row["ta"],
                "cc": row["cc"],
                "lr": row["lr"],
                "gr": row["gr"],
            })
        return examples

    def create_few_shot_prompt(self, examples: List[Dict[str, Any]], target_question: str, band: int) -> str:
        prompt = """You are simulating an IELTS test taker writing an essay. You must write at the EXACT same quality level as the provided examples.

CRITICAL: Match the writing quality, grammar errors, vocabulary level, and sentence complexity of the examples. Write match quality with the examples shown.

Study these authentic IELTS essays carefully and match their style:

"""
        for index, example in enumerate(examples, 1):
            prompt += f"Example {index}:\n"
            prompt += f"Question: {example['question']}\n\n"
            prompt += f"Essay: {example['essay']}\n\n"
            prompt += (
                f"Scores - Overall: {example['overall']}, Task Achievement: {example['ta']}, "
                f"Coherence & Cohesion: {example['cc']}, Lexical Resource: {example['lr']}, "
                f"Grammatical Range: {example['gr']}\n\n---\n\n"
            )
        prompt += f"""Now write an essay for this question. You MUST write at the same quality level as the examples above. Include similar:
- Grammar mistakes and errors
- Simple vocabulary and sentence structures
- Basic ideas and limited development
- Similar writing style and fluency level
- Same level of coherence and organization

Write like the examples shown.

IELTS FORMAT REQUIREMENTS:
- Write exactly 250-300 words
- Use proper IELTS Task 2 structure:
  1. INTRODUCTION (1 paragraph): Paraphrase the question and state your position/thesis
  2. BODY (2-3 paragraphs): Develop your main arguments with examples and explanations
  3. CONCLUSION (1 paragraph): Summarize your main points and restate your position

IMPORTANT: Write in PLAIN TEXT only. Do NOT use any markdown formatting, bold text (**), italics (*), headers (#), bullet points, or special formatting. Write as a normal IELTS essay would appear on paper.

Question: {target_question}

Essay:"""
        return prompt

    def generate_essay(self, prompt: str, band: int, feedback: Optional[str] = None,
                       current_essay: Optional[str] = None, current_score: Optional[float] = None,
                       target_band: Optional[float] = None) -> Optional[str]:
        token_limits = {5: 400, 6: 450, 7: 500, 8: 550, 9: 600}
        max_tokens = token_limits.get(band, 500)
        if feedback and current_essay and current_score is not None and target_band is not None:
            if current_score > target_band:
                action = f"DOWNGRADE the essay quality from Band {current_score} to Band {target_band}. Base this on the feedback."
            elif current_score < target_band:
                action = f"IMPROVE the essay quality from Band {current_score} to Band {target_band}. Base this on the feedback."
            else:
                action = f"MAINTAIN the current Band {target_band} quality while addressing the feedback."
            system_content = f"""You are rewriting an IELTS Task 2 essay. {action}

ESSAY TO REVISE:
{current_essay}

FEEDBACK TO ADDRESS:
{feedback}

Write a revised essay that addresses the feedback and achieves exactly Band {target_band} characteristics. Focus on the specific changes mentioned in the feedback.

IELTS FORMAT REQUIREMENTS:
- Write exactly 250-300 words
- Use proper IELTS Task 2 structure with introduction, body paragraphs, and conclusion.

IMPORTANT: Write in PLAIN TEXT only. Do NOT use markdown formatting."""
        else:
            system_content = """You are simulating an IELTS test taker writing a Task 2 essay. Write essays that match the EXACT quality level of the examples provided. Include grammar errors, simple vocabulary, and basic sentence structures. Do NOT write perfect essays - match the authentic student writing level shown in the examples.

IELTS FORMAT REQUIREMENTS:
- Write exactly 250-300 words
- Use proper IELTS Task 2 structure with introduction, body paragraphs, and conclusion.

IMPORTANT: Write in PLAIN TEXT only. Do NOT use markdown formatting."""
        band_desc = self.load_band_descriptions().get(str(band), {})
        if band_desc:
            desc_text = "\n".join(f"{key.upper()}: {value}" for key, value in band_desc.items())
            system_content += f"\n\nHere are the characteristics of a Band {band} essay:\n{desc_text}\n\nEnsure your essay matches these characteristics exactly."
        try:
            return self.complete_with_retry(system_content=system_content, user_content=prompt, max_tokens=max_tokens)
        except Exception as exc:
            print(f"Error generating essay: {exc}")
            traceback.print_exc()
            return None

    def extract_json_object(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def create_default_scores(self, feedback: str) -> Dict[str, Any]:
        return {
            "task_achievement": 6.0,
            "coherence_cohesion": 6.0,
            "lexical_resource": 6.0,
            "grammatical_range": 6.0,
            "overall": 6.0,
            "feedback": feedback,
            "word_count": None,
            "format_issues": "Scoring unavailable",
        }

    def score_essay(self, essay: str, question: str) -> Dict[str, Any]:
        band_descriptions = self.load_band_descriptions()
        system_content = """You are an expert IELTS examiner. Score this essay on the four IELTS criteria using the provided band descriptors.

ALSO evaluate IELTS format requirements:
- Word count should be 250-300 words (deduct points if significantly under/over)
- Must have clear 3-part structure: Introduction, Body paragraphs, Conclusion

Return your response in this exact JSON format:
{
    "task_achievement": 6.5,
    "coherence_cohesion": 6.0,
    "lexical_resource": 7.0,
    "grammatical_range": 6.5,
    "overall": 6.5,
    "feedback": "Brief explanation of the scoring including format adherence",
    "word_count": 275,
    "format_issues": "Any structure/format problems identified"
}

Use only these band scores: 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0
Overall score should be the average of the four criteria scores, rounded to nearest 0.5."""
        band_context = "IELTS Band Descriptors:\n"
        for band, criteria in band_descriptions.items():
            band_context += f"\nBAND {band}:\n"
            for criterion, description in criteria.items():
                band_context += f"{criterion.upper()}: {description}\n"
        user_content = f"""Question: {question}

Essay to score:
{essay}

{band_context}

Please score this essay according to IELTS criteria."""
        try:
            response = self.complete_with_retry(system_content=system_content, user_content=user_content, max_tokens=300)
            try:
                return self.extract_json_object(response)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON response: {response}")
                return self.create_default_scores("Scoring failed - used default scores")
        except Exception as exc:
            print(f"Error scoring essay: {exc}")
            return self.create_default_scores(f"Scoring error: {exc}")

    def generate_feedback(self, essay: str, question: str, current_scores: Dict[str, Any], target_band: float) -> str:
        band_descriptions = self.load_band_descriptions()
        current_overall = current_scores.get("overall", 0)
        target_band_key = str(int(target_band)) if target_band == int(target_band) else str(int(target_band + 0.5))
        target_descriptors = band_descriptions.get(target_band_key, band_descriptions.get("7", {}))
        if current_overall > target_band:
            action = "DOWNGRADE"
            direction = f"reduce the quality from Band {current_overall} to Band {target_band}"
            instruction = "Make the essay less sophisticated, introduce more errors, simplify vocabulary and sentence structures"
        elif current_overall < target_band:
            action = "IMPROVE"
            direction = f"improve the quality from Band {current_overall} to Band {target_band}"
            instruction = "Make the essay more sophisticated, reduce errors, enhance vocabulary and sentence complexity"
        else:
            action = "MAINTAIN"
            direction = f"maintain the current Band {target_band} level"
            instruction = "Keep the current quality level while making minor adjustments"
        system_content = f"""Provide specific, actionable feedback to {direction}.

ACTION REQUIRED: {action} the essay quality
INSTRUCTION: {instruction}

Current scores:
- Task Achievement: {current_scores.get('task_achievement', 'N/A')}
- Coherence & Cohesion: {current_scores.get('coherence_cohesion', 'N/A')}
- Lexical Resource: {current_scores.get('lexical_resource', 'N/A')}
- Grammatical Range: {current_scores.get('grammatical_range', 'N/A')}
- Overall: {current_scores.get('overall', 'N/A')}
- Word Count: {current_scores.get('word_count', 'N/A')}
- Format Issues: {current_scores.get('format_issues', 'None')}

Target Band {target_band} Requirements:
{json.dumps(target_descriptors, indent=2)}

Provide specific changes needed to {action} the essay to exactly Band {target_band}."""
        user_content = f"""Question: {question}

Current Essay:
{essay}

Please provide specific change recommendations to exactly reach Band {target_band}."""
        try:
            return self.complete_with_retry(system_content=system_content, user_content=user_content, max_tokens=400)
        except Exception as exc:
            return f"Could not generate feedback due to error: {exc}"

    def improve_essay_iteratively(self, question: str, target_band: int, max_iterations: int) -> List[Dict[str, Any]]:
        examples = self.get_examples_for_band(target_band)
        iterations = []
        current_essay = None
        feedback = None
        current_score = None
        print(f"Starting iterative improvement toward target band {target_band}")
        print(f"Question: {question}")
        print("=" * 80)
        for iteration in range(max_iterations):
            print(f"\n--- ITERATION {iteration + 1} ---")
            if iteration == 0:
                prompt = self.create_few_shot_prompt(examples, question, target_band)
                essay = self.generate_essay(prompt, target_band)
            elif current_essay and feedback and current_score is not None:
                improve_prompt = f"""Question: {question}

Please rewrite this essay to address the feedback and reach Band {target_band} level:
{feedback}
"""
                essay = self.generate_essay(improve_prompt, target_band, feedback, current_essay, current_score, target_band)
            else:
                print("No feedback available for improvement")
                break
            if not essay:
                print(f"Failed to generate essay for iteration {iteration + 1}")
                break
            print("Scoring essay...")
            scores = self.score_essay(essay, question)
            current_overall = scores.get("overall", 0)
            if iteration < max_iterations - 1 and abs(current_overall - target_band) > 0.5:
                print("Generating feedback...")
                feedback = self.generate_feedback(essay, question, scores, target_band)
            else:
                feedback = None
            iterations.append({
                "iteration": iteration + 1,
                "essay": essay,
                "scores": scores,
                "feedback": feedback,
                "target_reached": abs(current_overall - target_band) <= 0.5,
            })
            print(f"Essay: {essay[:100]}...")
            print(
                f"Scores: TA={scores.get('task_achievement', 'N/A')}, "
                f"CC={scores.get('coherence_cohesion', 'N/A')}, "
                f"LR={scores.get('lexical_resource', 'N/A')}, "
                f"GR={scores.get('grammatical_range', 'N/A')}, Overall={current_overall}"
            )
            if feedback:
                action = "IMPROVE" if current_overall < target_band else "DOWNGRADE" if current_overall > target_band else "MAINTAIN"
                print(f"Next iteration will: {action} (Current: {current_overall} -> Target: {target_band})")
                print(f"Feedback for next iteration: {feedback}")
            if abs(current_overall - target_band) <= 0.5:
                print(f"Target band {target_band} reached in iteration {iteration + 1}!")
                break
            current_essay = essay
            current_score = current_overall
            time.sleep(1)
        return iterations


def build_output_filename(bands: List[int]) -> str:
    sorted_bands = sorted(bands)
    min_band = min(sorted_bands)
    max_band = max(sorted_bands)
    if len(sorted_bands) == 1:
        band_str = str(sorted_bands[0])
    elif sorted_bands == list(range(min_band, max_band + 1)):
        band_str = f"{min_band}-{max_band}"
    else:
        band_str = "-".join(map(str, sorted_bands))
    timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
    return f"band_{band_str}_{timestamp}.csv"


def save_results(all_results: List[Dict[str, Any]], bands: List[int], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_essays = [{"Question": row["question"], "Essay": row["essay"], "Overall": row["target_band"]} for row in all_results]
    output_path = output_dir / build_output_filename(bands)
    pd.DataFrame(final_essays).to_csv(output_path, index=False)
    print(f"\nSaved {len(final_essays)} essays to {output_path}")
    return output_path


def print_summary(all_results: List[Dict[str, Any]], bands: List[int]) -> None:
    print("\nCOMBINED SUMMARY RESULTS:")
    print(f"Target bands: {', '.join(map(str, bands))}")
    print(f"Total essays processed: {len(all_results)}")
    for band in sorted(bands):
        band_results = [row for row in all_results if row["target_band"] == band]
        if not band_results:
            continue
        final_scores = [row["scores"].get("overall", 0) for row in band_results]
        success_count = sum(1 for score in final_scores if abs(score - band) <= 0.5)
        avg_score = sum(final_scores) / len(final_scores)
        print(f"\nBand {band} Results:")
        print(f"  Essays: {len(band_results)}")
        print(f"  Success rate: {success_count}/{len(band_results)} ({success_count / len(band_results) * 100:.1f}%)")
        print(f"  Average final score: {avg_score:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic IELTS essays using OpenAI or Claude")
    parser.add_argument("bands", type=parse_bands, help="Target band score(s), e.g. 7 or 5,6,7")
    parser.add_argument("--provider", choices=["openai", "claude"], default=os.getenv("SYNTHETIC_PROVIDER", "openai"))
    parser.add_argument("--num-essays", type=int, default=env_int("SYNTHETIC_NUM_ESSAYS", 1))
    parser.add_argument("--max-iterations", type=int, default=env_int("SYNTHETIC_MAX_ITERATIONS", 1))
    parser.add_argument("--asset-dir", type=Path, default=Path(os.getenv("SYNTHETIC_ASSET_DIR", DEFAULT_ASSET_DIR)))
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("SYNTHETIC_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--model", default=None, help="Override selected provider model")
    parser.add_argument("--temperature", type=float, default=env_float("OPENAI_TEMPERATURE", env_float("ANTHROPIC_TEMPERATURE", 0.7)))
    parser.add_argument("--max-retries", type=int, default=env_int("SYNTHETIC_MAX_RETRIES", env_int("OPENAI_MAX_RETRIES", env_int("ANTHROPIC_MAX_RETRIES", 3))))
    parser.add_argument("--retry-delay", type=float, default=env_float("SYNTHETIC_RETRY_DELAY", env_float("OPENAI_RETRY_DELAY", env_float("ANTHROPIC_RETRY_DELAY", 2.0))))
    parser.add_argument("--seed", type=int, default=env_int("SYNTHETIC_RANDOM_SEED", 42))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv_or_exit()
    random.seed(args.seed)
    provider = build_provider(args.provider, args.model)
    generator = SyntheticIELTSGenerator(args.asset_dir, provider, args.temperature, args.max_retries, args.retry_delay)
    try:
        _, questions_df = generator.load_data()
    except Exception as exc:
        print(f"Error loading generation assets: {exc}")
        sys.exit(1)
    print("Running iterative improvement mode:")
    print(f"- Provider: {provider.name}")
    print(f"- Model: {provider.model}")
    print(f"- Target bands: {', '.join(map(str, args.bands))}")
    print(f"- Number of essays per band: {args.num_essays}")
    print(f"- Max iterations per essay: {args.max_iterations}")
    print(f"- Asset directory: {args.asset_dir}")
    print(f"- Output directory: {args.output_dir}")
    all_results = []
    essay_counter = 1
    for band in args.bands:
        print(f"\n{'=' * 80}\nPROCESSING TARGET BAND {band}\n{'=' * 80}")
        for essay_num in range(args.num_essays):
            print(f"\n{'=' * 60}\nPROCESSING ESSAY {essay_counter} (Band {band}, Essay {essay_num + 1}/{args.num_essays})\n{'=' * 60}")
            question = questions_df.sample(n=1, random_state=random.randint(0, 2**32 - 1)).iloc[0]["question"]
            iterations = generator.improve_essay_iteratively(question, band, args.max_iterations)
            if iterations:
                final_iteration = iterations[-1]
                all_results.append({
                    "essay_number": essay_counter,
                    "target_band": band,
                    "question": question,
                    "essay": final_iteration["essay"],
                    "scores": final_iteration["scores"],
                })
            essay_counter += 1
    if all_results:
        save_results(all_results, args.bands, args.output_dir)
        print_summary(all_results, args.bands)
    else:
        print("No essays were generated.")


if __name__ == "__main__":
    main()
