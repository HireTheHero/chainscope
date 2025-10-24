#!/usr/bin/env python3
"""E.g. run:

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_historical \
    --model_id "anthropic/claude-3.7-sonnet:thinking" \
    --open_router \
    --max_retries=1 \
    --prefix=1 \
    --verbose

Or:

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_historical \
    --model_id "qwen/qwen-2.5-72b-instruct" \
    --max_retries=3 \
    --verbose

Or (with temperature and 2024 Putnam problems):

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_2024 \
    --model_id "anthropic/claude-3.7-sonnet:thinking" \
    --open_router \
    --temperature=0.3 \
    --max_retries=1 \
    --prefix=1 \
    --epochs=2 \
    --verbose

Or (for the specific NeurIPS Sonnet non-thinking experiment):

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_neurips_sonnet_nonthinking \
    --model_id "anthropic/claude-3.7-sonnet" \
    --open_router \
    --epochs=2 \
    --max_retries=1 \
    --verbose

"""

import asyncio
import json
import logging
import os
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from vllm import LLM
from vllm import SamplingParams as VLLMSamplingParams

from chainscope.api_utils.deepseek_utils import (
    DeepSeekBatchProcessor,
    DeepSeekRateLimiter,
)
from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.api_utils import anthropic_utils  # import ANBatchProcessor
from chainscope.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathQsDataset,
    MathQuestion,
    MathResponse,
    QuestionResponseId,
    SamplingParams,
)
from chainscope.utils import MODELS_MAP, is_instruct_model, make_chat_prompt


class DatasetType(StrEnum):
    PUTNAM_HISTORICAL = "putnam_historical"  # For the historical dataset
    PUTNAM_2024 = "putnam_2024"  # For 2024 problems
    PUTNAM_NEURIPS_SONNET_NONTHINKING = "putnam_neurips_sonnet_nonthinking" # For the specific NeurIPS experiment

    @property
    def dataset_id(self) -> str:
        """Get the dataset ID for this type."""
        match self:
            case DatasetType.PUTNAM_HISTORICAL:
                return "filtered_putnambench"
            case DatasetType.PUTNAM_2024:
                return "ten_putnam_2024_problems"
            case DatasetType.PUTNAM_NEURIPS_SONNET_NONTHINKING:
                return "putnam_neurips_sonnet_nonthinking_experiment"

    @property
    def description(self) -> str:
        """Get the dataset description for this type."""
        match self:
            case DatasetType.PUTNAM_HISTORICAL:
                return "Historical Putnam Competition Problems"
            case DatasetType.PUTNAM_2024:
                return "Putnam Competition Problems 2024"
            case DatasetType.PUTNAM_NEURIPS_SONNET_NONTHINKING:
                return "Putnam Problems from NeurIPS Sonnet Non-Thinking Experiment"

    @property
    def yaml_path(self) -> str:
        """Get the YAML file path for this dataset type."""
        match self:
            case DatasetType.PUTNAM_HISTORICAL:
                return "d/putnam2/minimal_fork_of_putnambench_with_clear_answers.yaml"
            case DatasetType.PUTNAM_2024:
                return "d/putnam2/ten_putnam_2024_problems.yaml"
            case DatasetType.PUTNAM_NEURIPS_SONNET_NONTHINKING:
                # This path should be relative to the workspace root if the script is run from there,
                # or it needs to be an absolute path or adjusted based on execution context.
                # For now, assuming it's relative to the workspace root as per user's notebook file.
                return "chainscope/data/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/putnam_neurips_experiment_claude_sonnet_nonthinking.yaml"


def load_putnam_results_as_df(yaml_path: Path) -> pd.DataFrame:
    """Load Putnam results from YAML into a pandas DataFrame."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def get_putnam_responses_vllm(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    sampling_params: SamplingParams,
) -> list[tuple[QuestionResponseId, str, str | None]]:
    """Generate responses using VLLM for Putnam problems.
    
    This is a simplified version that doesn't use FSP since Putnam problems
    don't use the same dataset structure as IPHR.
    """
    # Initialize vLLM engine
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
    )
    
    # Convert our sampling params to vLLM format
    vllm_params = VLLMSamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=sampling_params.max_new_tokens,
    )
    
    # Prepare prompts
    prompt_texts = []
    q_resp_ids = []
    
    for q_resp_id, prompt in tqdm(prompts, desc="Preparing prompts"):
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=llm.get_tokenizer(),  # type: ignore
            )
        else:
            input_str = prompt
        
        prompt_texts.append(input_str)
        q_resp_ids.append(q_resp_id)
    
    # Generate responses using vLLM
    logging.info(f"Generating {len(prompt_texts)} responses")
    all_outputs = llm.generate(prompt_texts, vllm_params, use_tqdm=True)
    logging.info(f"Generated {len(all_outputs)} responses")
    
    # Format responses
    responses: list[tuple[QuestionResponseId, str, str | None]] = []
    for q_resp_id, output in tqdm(
        zip(q_resp_ids, all_outputs), desc="Processing responses", total=len(q_resp_ids)
    ):
        generated_text = output.outputs[0].text
        responses.append((q_resp_id, generated_text, None))
    
    return responses


def get_putnam_responses_tl(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    sampling_params: SamplingParams,
    local_gen_seed: int,
) -> list[tuple[QuestionResponseId, str, str | None]]:
    """Generate responses using TransformerLens for Putnam problems.
    
    This is a simplified version that doesn't use FSP since Putnam problems
    don't use the same dataset structure as IPHR.
    """   
    # Set TransformerLens seed for reproducible local generation
    HookedTransformerConfig.set_seed_everywhere(
        None,  # type: ignore
        local_gen_seed,
    )
    
    # Initialize TransformerLens model
    # Check if model_id is a local path (starts with /, ./, or ../)
    is_local_path = model_id.startswith('/') or model_id.startswith('./') or model_id.startswith('../')
    
    if is_local_path:
        logging.info(f"Loading model from local path: {model_id}")
        # Read config to get the model architecture
        config_path = Path(model_id) / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Try to infer the official model name from the config
        # For Llama models, look at the model_type
        model_type = config.get("model_type", "")
        architectures = config.get("architectures", [])
        
        # Map to official HF model name for TransformerLens
        # This is a heuristic - adjust as needed
        if "llama" in model_type.lower() or any("Llama" in arch for arch in architectures):
            # Check if it's Llama 3.1 based on vocab size or other indicators
            # Llama 3.1 has 128256 vocab size
            vocab_size = config.get("vocab_size", 0)
            if vocab_size == 128256:
                # Assume it's Llama 3.1 - use the official name
                official_name = "meta-llama/Llama-3.1-70B"
                logging.info(f"Detected Llama 3.1 model, using official name: {official_name}")
            else:
                raise ValueError(f"Could not determine official model name for local path: {model_id}")
        else:
            raise ValueError(f"Unsupported model type '{model_type}' for local path loading")
        
        # Set cache dir to the parent of the local model
        os.environ["HF_HOME"] = str(Path(model_id).parent)
        os.environ["TRANSFORMERS_CACHE"] = str(Path(model_id).parent)
        
        model = HookedTransformer.from_pretrained(
            model_name=official_name,
            device="cuda",
            local_files_only=True,
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name=model_id,
            device="cuda",
        )
    assert model.tokenizer is not None, "Tokenizer is not initialized"
    
    # Prepare prompts and generate responses
    responses: list[tuple[QuestionResponseId, str, str | None]] = []
    
    for q_resp_id, prompt in tqdm(prompts, desc="Generating responses"):
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=model.tokenizer,  # type: ignore
            )
        else:
            input_str = prompt
        
        # Tokenize input
        tokens = model.to_tokens(input_str, prepend_bos=True).to(model.cfg.device)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1
        
        # Generate the full sequence at once
        with torch.inference_mode():
            generated = model.generate(
                tokens,
                max_new_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                return_type="tokens",
                verbose=False,
            )
            assert isinstance(generated, torch.Tensor)
            assert generated.ndim == 2
        
        # Convert output tokens to text
        generated_text = model.tokenizer.batch_decode(
            generated[:, tokens.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        assert isinstance(generated_text, str)
        
        responses.append((q_resp_id, generated_text, None))
    
    return responses


def convert_putnam_to_local_format(
    dataset: MathQsDataset,
    preamble: str = "",
    prefix: Optional[int] = None,
    epochs: int = 1,
) -> list[tuple[QuestionResponseId, str]]:
    """Convert Putnam dataset to format expected by local generation functions.
    
    Args:
        dataset: Putnam dataset
        preamble: Preamble text to add before each problem
        prefix: Only process first N problems if specified
        epochs: Number of epochs to generate
        
    Returns:
        List of (QuestionResponseId, prompt) tuples
    """
    questions = dataset.questions[:prefix] if prefix else dataset.questions
    prompts = []
    
    for epoch in range(epochs):
        for question in questions:
            # Create question name with epoch if > 1
            if epochs > 1:
                question_name = f"{question.name}_attempt_{epoch + 1}"
            else:
                question_name = question.name
            
            # Create a QuestionResponseId for this question-response pair
            q_resp_id = QuestionResponseId(
                qid=question_name,
                uuid=str(uuid.uuid4())
            )
            
            prompt = f"{preamble}{question.problem}"
            prompts.append((q_resp_id, prompt))
    
    return prompts


def convert_local_results_to_putnam(
    results: list[tuple[QuestionResponseId, str, str | None]],
    dataset: MathQsDataset,
    model_id: str,
    epochs: int = 1,
) -> CotResponses:
    """Convert local generation results back to Putnam format.
    
    Args:
        results: Results from local generation (QuestionResponseId, response, fsp)
        dataset: Original Putnam dataset
        model_id: Model ID used for generation
        epochs: Number of epochs processed
        
    Returns:
        CotResponses object in Putnam format
    """
    responses_by_qid = {}
    
    # Group results by question name
    for q_resp_id, response, fsp in results:
        question_name = q_resp_id.qid
        if not response:
            continue
            
        # For multiple epochs, handle attempt numbering
        if epochs > 1:
            # Extract base name and attempt number if present
            if "_attempt_" in question_name:
                base_name = question_name.rsplit("_attempt_", 1)[0]
                attempt_num = int(question_name.rsplit("_attempt_", 1)[1])
            else:
                base_name = question_name
                attempt_num = 1
        else:
            base_name = question_name
            attempt_num = 1
        
        # Find the original question
        original_question = None
        for q in dataset.questions:
            if q.name == base_name:
                original_question = q
                break
        
        if original_question is None:
            logging.warning(f"Could not find original question for {base_name}")
            continue
        
        # Initialize dict for this question if it doesn't exist
        if question_name not in responses_by_qid:
            responses_by_qid[question_name] = {}
            
        # Add this response with a unique ID
        responses_by_qid[question_name][str(uuid.uuid4())[:8]] = MathResponse(
            name=question_name,
            problem=original_question.problem,
            solution=original_question.solution,
            model_thinking=None,  # Local generation doesn't separate thinking
            model_answer=[response],  # Store as single response
        )
    
    # Sort responses by question name
    def sort_key(name: str) -> tuple:
        # Handle both formats: putnam_2024_a1 and putnam_2024_a1_attempt_1
        parts = name.split('_')
        if len(parts) >= 4:  # Has problem number
            year = int(parts[1])
            prob_type = parts[2][0]  # 'a' or 'b'
            prob_num = int(parts[2][1])
            attempt = int(parts[-1]) if len(parts) > 4 else 0
            return (year, prob_type, prob_num, attempt)
        return (0, '', 0, 0)  # Fallback for unexpected formats

    sorted_responses = dict(sorted(responses_by_qid.items(), key=lambda x: sort_key(x[0])))

    return CotResponses(
        responses_by_qid=sorted_responses,
        model_id=model_id,
        instr_id="instr-v0",
        ds_params=dataset.params,
        sampling_params=DefaultSamplingParams(),
    )


def create_putnam_dataset(dataset_type: DatasetType) -> MathQsDataset:
    """Create a MathQsDataset based on the dataset type.
    
    Args:
        dataset_type: Type of dataset to create
        
    Returns:
        A MathQsDataset containing the problems for the specified type
    """
    # Load and convert to DataFrame
    df = load_putnam_results_as_df(Path(dataset_type.yaml_path))
    
    # Sort problems by year and type
    df = df.sort_values(
        by="problem_name",
        key=lambda x: pd.Series(
            [
                # Extract year and problem type (e.g. 'a1', 'b2')
                (int(name.split("_")[1]), name.split("_")[2])
                for name in x
            ]
        ).map(
            lambda t: (
                {
                    "a1": 0,
                    "b1": 1,
                    "a2": 2,
                    "b2": 3,
                    "a3": 4,
                    "b3": 5,
                    "a4": 6,
                    "b4": 7,
                    "a5": 8,
                    "b5": 9,
                    "a6": 10,
                    "b6": 11,
                }[t[1]],
                -t[0],
            )
        ),
    )

    return MathQsDataset(
        questions=[
            MathQuestion(
                name=row["problem_name"],
                problem=row["informal_statement"],
                solution=row["informal_solution"],
            )
            for _, row in df.iterrows()
        ],
        params=MathDatasetParams(
            description=dataset_type.description,
            id=dataset_type.dataset_id,
            pre_id=None,
        ),
    )


def create_processor(
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    temperature: float = 0.0,
    force_open_router: bool = False,
):
    """Create the appropriate processor based on the model ID."""

    def get_tuple_or_str_response(
        response: tuple[str, str] | str, other: Any
    ) -> tuple[str | None, str]:
        logging.info(f"Inner response: {response}")

        if isinstance(response, tuple):
            assert (
                len(response) == 2
            ), f"Expected tuple of length 2, got {len(response)}"
            return response
        else:
            return (None, response)

    if anthropic_utils.ANBatchProcessor.is_model_supported(model_id) and not force_open_router:
        # Anthropic processor
        logging.info(f"Using Anthropic model {model_id}")
        rate_limiter = None
        if max_parallel is not None:
            rate_limiter = ORRateLimiter(
                requests_per_interval=max_parallel,
                interval_seconds=1,
            )
        return anthropic_utils.ANBatchProcessor[MathQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            # If _32k budget then do 1.25* that many tokens etc:
            max_new_tokens=32_000 if "_" not in model_id else int(int(model_id.split("_")[-1][:-1]) * 1.25),
            temperature=temperature,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
        )
    elif DeepSeekBatchProcessor.is_model_supported(model_id) and not force_open_router:
        return DeepSeekBatchProcessor[MathQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=8_192,
            temperature=temperature,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
            # NOTE: Only used when thinking is also returned
            format_thinking=lambda thinking,
            answer: f"**WORKING**: {thinking.lstrip()}\n\n**ANSWER**: {answer.lstrip()}",
        )
    else:
        # OpenRouter processor
        logging.info(f"Using OpenRouter model {model_id}")
        rate_limiter = None
        if max_parallel is not None:
            rate_limiter = ORRateLimiter(
                requests_per_interval=max_parallel,
                interval_seconds=1,
            )
        return ORBatchProcessor[MathQuestion, tuple[str | None, str]](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=32_000,
            temperature=temperature,
            process_response=get_tuple_or_str_response,
            rate_limiter=rate_limiter,
        )


async def generate_rollouts_local(
    dataset: MathQsDataset,
    model_id: str,
    api: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 2000,
    prefix: Optional[int] = None,
    preamble: str = "",
    epochs: int = 1,
    model_id_for_fsp: Optional[str] = None,
    fsp_size: int = 5,
    fsp_seed: int = 42,
    local_gen_seed: int = 42,
) -> CotResponses:
    """Generate rollouts using local models (VLLM or TTL).
    
    Args:
        dataset: Putnam dataset
        model_id: Model ID for generation
        api: Local API to use ("vllm" or "ttl")
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        prefix: Only process first N problems if specified
        preamble: Preamble text to add before each problem
        epochs: Number of times to process each problem
        model_id_for_fsp: Model ID for few-shot prompting (optional)
        fsp_size: Number of few-shot examples
        fsp_seed: Seed for few-shot example selection
        local_gen_seed: Seed for local generation
        
    Returns:
        CotResponses object
    """
    logging.info(f"Using local generation with {api} for model {model_id}")
    
    # Convert model ID using MODELS_MAP
    model_id = MODELS_MAP.get(model_id, model_id)
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    
    # Convert Putnam data to local format (already handles epochs)
    all_prompts = convert_putnam_to_local_format(dataset, preamble, prefix, epochs)
    
    if not all_prompts:
        logging.info("No prompts to process")
        return CotResponses(
            responses_by_qid={},
            model_id=model_id,
            instr_id="instr-v0",
            ds_params=dataset.params,
            sampling_params=sampling_params,
        )
    
    # Generate responses using local model
    # Note: FSP is not currently supported for Putnam problems due to different data structures
    if model_id_for_fsp is not None:
        logging.warning("Few-shot prompting (--model-id-for-fsp) is not currently supported for Putnam problems")
    
    if api == "vllm":
        results = get_putnam_responses_vllm(
            prompts=all_prompts,
            model_id=model_id,
            sampling_params=sampling_params,
        )
    else:  # ttl
        results = get_putnam_responses_tl(
            prompts=all_prompts,
            model_id=model_id,
            sampling_params=sampling_params,
            local_gen_seed=local_gen_seed,
        )
    
    if not results:
        logging.warning("No results generated")
        return CotResponses(
            responses_by_qid={},
            model_id=model_id,
            instr_id="instr-v0",
            ds_params=dataset.params,
            sampling_params=sampling_params,
        )
    
    # Convert results back to Putnam format
    return convert_local_results_to_putnam(
        results=results,
        dataset=dataset,
        model_id=model_id,
        epochs=epochs,
    )


async def generate_rollouts(
    dataset: MathQsDataset,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    temperature: float = 0.0,
    prefix: Optional[int] = None,
    force_open_router: bool = False,
    preamble: str = "",
    epochs: int = 1,
) -> CotResponses:
    """Generate rollouts for each problem in the dataset.
    
    Args:
        epochs: Number of times to process each problem. If > 1, will generate multiple responses per problem.
    """
    processor = create_processor(
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        temperature=temperature,
        force_open_router=force_open_router,
    )

    # Prepare questions for processing
    questions = dataset.questions[:prefix] if prefix else dataset.questions

    logging.warning("USING THINK STEP-BY-STEP PREFIX! ('preamble')")
    
    # Create batch items for all questions × epochs
    batch_items = []
    for _ in range(epochs):
        batch_items.extend([
            (
                q,
                f"{preamble}{q.problem}",
            )
            for q in questions
        ])
    
    # Process all questions in a single batch
    logging.info(f"Processing {len(batch_items)} problems")
    results = await processor.process_batch(batch_items)

    # Process all questions in batch
    responses_by_qid = {}
    
    # Group responses by question
    for batch_idx, ((question, _), (_, thinking_and_answer)) in enumerate(zip(batch_items, results)):
        if thinking_and_answer is None or thinking_and_answer[-1] is None:
            logging.warning(
                f"Skipping failed response for {question.name} {thinking_and_answer=}"
            )
            continue

        thinking, answer = thinking_and_answer
        
        # For multiple epochs, append attempt number to question name:
        if epochs > 1:
            # Vibe code slop but w/e:
            attempt_number = batch_idx // len(dataset.questions[:prefix] if prefix else dataset.questions) + 1
            question_name = f"{question.name}_attempt_{attempt_number}"
        else:
            question_name = question.name
        
        # Initialize dict for this question if it doesn't exist
        if question_name not in responses_by_qid:
            responses_by_qid[question_name] = {}
            
        # Add this response with a unique ID
        responses_by_qid[question_name][str(uuid.uuid4())[:8]] = MathResponse(
            name=question_name,
            problem=question.problem,
            solution=question.solution,
            model_thinking=thinking,
            model_answer=[answer],  # Unsplit
        )

    # Sort responses by question name after all are collected
    def sort_key(name: str) -> tuple:
        # Handle both formats: putnam_2024_a1 and putnam_2024_a1_attempt_1
        parts = name.split('_')
        if len(parts) >= 4:  # Has problem number
            year = int(parts[1])
            prob_type = parts[2][0]  # 'a' or 'b'
            prob_num = int(parts[2][1])
            attempt = int(parts[-1]) if len(parts) > 4 else 0
            return (year, prob_type, prob_num, attempt)
        return (0, '', 0, 0)  # Fallback for unexpected formats

    sorted_responses = dict(sorted(responses_by_qid.items(), key=lambda x: sort_key(x[0])))

    return CotResponses(
        responses_by_qid=sorted_responses,
        model_id=model_id,
        instr_id="instr-v0",
        ds_params=dataset.params,
        sampling_params=DefaultSamplingParams(),
    )


@click.command()
@click.option(
    "--dataset_type",
    "-d",
    type=click.Choice([t.value for t in DatasetType], case_sensitive=False),
    required=True,
    help="Type of dataset being processed",
)
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3-opus",
    help="Model ID for generating rollouts (OpenRouter or DeepSeek model)",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=1,
    help="Maximum retries for failed requests",
)
@click.option(
    "--max_parallel",
    "-p",
    type=int,
    default=None,
    help="Maximum number of parallel requests",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.0,
    help="Sampling temperature for the model",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=1,
    help="Number of times to process each problem",
)
@click.option(
    "--prefix",
    "-prefix",
    type=int,
    default=None,
    help="Only process the first N problems",
)
@click.option(
    "--preamble",
    type=str,
    default="Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n",
    help="Preamble text to add before each problem",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--open_router",
    is_flag=True,
    help="Force using OpenRouter even for DeepSeek models",
)
@click.option(
    "--api",
    type=click.Choice(["vllm", "ttl"]),
    default=None,
    help="Use local API for generation (vllm or ttl)",
)
@click.option(
    "--top-p",
    type=float,
    default=0.9,
    help="Top-p sampling parameter for local generation",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=2000,
    help="Maximum new tokens to generate for local generation",
)
@click.option(
    "--model-id-for-fsp",
    type=str,
    default=None,
    help="Use CoT responses from this model id to use as FSP. Only used if generating responses for a base model.",
)
@click.option(
    "--fsp-size",
    type=int,
    default=5,
    help="Size of FSP to use for generation with --model-id-for-fsp",
)
@click.option(
    "--fsp-seed",
    type=int,
    default=42,
    help="Seed for FSP selection",
)
@click.option(
    "--local-gen-seed",
    type=int,
    default=42,
    help="Seed for local generation",
)
def main(
    dataset_type: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    temperature: float,
    epochs: int,
    prefix: Optional[int],
    verbose: bool,
    open_router: bool,
    preamble: str,
    api: Optional[str],
    top_p: float,
    max_new_tokens: int,
    model_id_for_fsp: Optional[str],
    fsp_size: int,
    fsp_seed: int,
    local_gen_seed: int,
):
    """Generate rollouts for Putnam problems using OpenRouter or DeepSeek models."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Convert dataset type string to enum
    dataset_type_enum = DatasetType(dataset_type)

    # Create dataset directly based on type
    dataset = create_putnam_dataset(dataset_type_enum)

    # Generate rollouts
    if api is not None:
        # Use local generation
        results = asyncio.run(
            generate_rollouts_local(
                dataset=dataset,
                model_id=model_id,
                api=api,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                prefix=prefix,
                preamble=preamble,
                epochs=epochs,
                model_id_for_fsp=model_id_for_fsp,
                fsp_size=fsp_size,
                fsp_seed=fsp_seed,
                local_gen_seed=local_gen_seed,
            )
        )
    else:
        # Use cloud APIs
        results = asyncio.run(
            generate_rollouts(
                dataset=dataset,
                model_id=model_id,
                preamble=preamble,
                max_retries=max_retries,
                max_parallel=max_parallel,
                temperature=temperature,
                epochs=epochs,
                prefix=prefix,
                force_open_router=open_router,
            )
        )

    # Save results
    for i in range(0, 100):
        output_path = results.get_path(
            f"_v{i}" + (f"_prefix_{prefix}" if prefix else "")
        )
        if not os.path.exists(output_path):
            break

    saved_path = results.save(path=output_path)
    logging.info(f"Saved rollouts to {saved_path}")


if __name__ == "__main__":
    main()
