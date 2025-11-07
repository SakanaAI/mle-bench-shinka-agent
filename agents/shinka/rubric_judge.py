from functools import wraps
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
import random

import pandas as pd
from shinka.llm import LLMClient

RUBRIC_SYSTEM_PROMPT = """You are a Kaggle Grandmaster who is reviewing \
a machine learning implementation for a kaggle problem.
You will be given code and a rubric that describes \
a desired quality derived from machine learning best practices.
You will need to judge the implementation based on the rubric."""

RUBRIC_PROMPT = """
Here is the problem statement:
<problem_statement>

{task_description}

<end_of_problem_statement>

---

Here is the proposed code for the problem:

```python
{code}
```

Here is the rubric to judge the implementation:

{rubric}

Judge whether the code implements the rubric. 
Given that only executed lines are provided, if a functionality is not present, it means it is either not implemented or not executed.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, briefly discuss your assessment of the implementation, expected outcome and use the rubric to assess the correctness of the implementation.

In <JSON>, provide the assessment in JSON format with the following fields:
- "Pass" (bool): Whether the code implements the rubric.
- "Thoughts" (str): Your thoughts on the implementation and the rubric.
- "Improvements" (str): Suggestions for improvements to the implementation. Use only when the implementation does not pass the rubric.

Be concise and objective.
"""


# taken from ai-ai conference repo
def extract_between(
    content: str,
    start: str = "<json>",
    end: str = "</json>",
    return_dict: bool = True,
    fallback: bool = False,
) -> str | dict | None:
    """Extract text from between start and end tags.

    Args:
        content (str): The input string containing CUDA code

    Returns:
        str: The extracted text, or None if no text is found
    """
    match = re.search(f"{start}\s*(.*?)\s*{end}", content, re.DOTALL)
    if match:
        matched_str = match.group(1).strip()
        if return_dict:
            return json.loads(matched_str)
        else:
            return matched_str

    # Extracts any block between ``` and ```
    if fallback:
        match = re.search("```\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            matched_str = match.group(1).strip()
            if return_dict:
                return json.loads(matched_str)
            else:
                return matched_str
    return "none"


def retry(
    max_retries: int,
    message: str = "Failed to get result from LLM, retrying...",
    trigger_retry_fn: Callable | None = None,
    *,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """
    Decorator to retry a function up to max_retries times,
    using exponential backoff between attempts.
    Args:
        max_retries: Maximum number of attempts (including the first).
        message: Message to log when retrying.
        trigger_retry_fn: Function that receives the result and returns True
            if a retry should be attempted (e.g. when result is None).
            Defaults to retrying on None.
        initial_delay: Base sleep (seconds) before the second attempt.
        backoff_factor: Multiplier applied to the delay after each attempt.
        max_delay: Upper bound for the delay (seconds).
        jitter: When True, apply "full jitter" by sleeping a random
            amount in [0, computed_delay] to reduce thundering herds.
    """

    def default_trigger_retry_fn(_result) -> bool:
        return _result is None

    if trigger_retry_fn is None:
        trigger_retry_fn = default_trigger_retry_fn

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            delay = float(max(0.0, initial_delay))
            result = None
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    print(f"[retry] ({f.__name__}) Attempt {attempt}/{max_retries}")
                    result = f(*args, **kwargs)

                    should_retry = trigger_retry_fn(result)
                    if not should_retry:
                        return result

                    # If should retry due to result and attempts remain
                    if attempt < max_retries:
                        # Compute exponential backoff with optional jitter
                        sleep_for = min(max_delay, delay)
                        if jitter:
                            sleep_for = random.uniform(0, sleep_for)
                        print(
                            f"{message} Next retry in {sleep_for:.2f}s (result-triggered)"
                        )
                        time.sleep(max(0.0, sleep_for))
                        delay = min(max_delay, delay * max(1.0, backoff_factor))
                        continue
                    else:
                        # No attempts left; break out and return the last result below
                        break

                except Exception as e:
                    last_exception = e
                    print(f"Exception occurred: {e}")
                    if attempt < max_retries:
                        sleep_for = min(max_delay, delay)
                        if jitter:
                            sleep_for = random.uniform(0, sleep_for)
                        print(
                            f"{message} Next retry in {sleep_for:.2f}s (exception-triggered)"
                        )
                        time.sleep(max(0.0, sleep_for))
                        delay = min(max_delay, delay * max(1.0, backoff_factor))
                        continue
                    else:
                        # Exhausted all attempts with exception: re-raise
                        raise last_exception

            print(
                f"Failed to get result from LLM after {max_retries} attempts; returning last result: {result}"
            )
            return result

        return wrapper

    return decorator


class KaggleRubricJudge:
    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.llm_client = LLMClient(
            model_names=model_name,
            temperatures=temperature,
            max_tokens=max_tokens,
        )

    def _query_judge_rubric(self, formatted_prompt: str) -> dict | None:
        # Query the LLM
        try:
            result = self.llm_client.query(
                msg=formatted_prompt, system_msg=RUBRIC_SYSTEM_PROMPT
            )
        except Exception:
            return None

        if result is None:
            # If query failed, return False as a safe default
            return None

        # First try to extract JSON from ```json``` blocks
        try:
            response_data = extract_between(
                result.content,
                start="```json",
                end="```",
                return_dict=True,
                fallback=False,
            )
        except Exception as e:
            raise RuntimeError(f"Error parsing JSON response from LLM: {str(e)}") from e
        if not isinstance(response_data, dict):
            return None

        return response_data

    def judge_rubric(
        self,
        task_description: str,
        code: str,
        rubric: str,
        max_retries: int = 3,
    ) -> dict:
        """Judge whether the code implements the rubric.

        Args:
            code: The code implementing the experiment
            rubric: The rubric to judge against

        Returns:
            bool: True if the code passes the rubric, False otherwise
        """
        formatted_prompt = RUBRIC_PROMPT.format(
            task_description=task_description,
            rubric=rubric,
            code=code,
        )

        query_with_retry_fn = retry(
            trigger_retry_fn=lambda x: x is None,
            message="Failed to judge rubric, retrying...",
            max_retries=max_retries,
        )(self._query_judge_rubric)
        try:
            response_data = query_with_retry_fn(formatted_prompt)
        except Exception as e:
            return {"Pass": False, "Thoughts": f"Query failed: {e}", "Improvements": ""}
        if response_data is None:
            return {"Pass": False, "Thoughts": "Query failed", "Improvements": ""}
        return response_data

    def judge_all_rubrics_dep(
        self,
        task_description: str,
        code: str,
        rubrics: list[str],
    ) -> list[bool]:
        results = []
        for rubric in rubrics:
            rubric_output = self.judge_rubric(task_description, code, rubric)
            rubric_output["Rubric"] = rubric
            results.append(rubric_output)
        # Return a pandas dataframe with the results
        df = pd.DataFrame(results)
        return df, df["Pass"].mean()

    def judge_all_rubrics(
        self,
        task_description: str,
        code: str,
        rubrics: list[str],
    ) -> list[bool]:
        max_workers = len(rubrics)

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(self.judge_rubric, task_description, code, rubric): rubric
                for rubric in rubrics
            }
            for fut in as_completed(futs):
                out = fut.result()
                rubric = futs[fut]
                out["Rubric"] = rubric
                results.append(out)
        df = pd.DataFrame(results)
        return df, df["Pass"].sum()
