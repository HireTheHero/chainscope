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

Or (with local HuggingFace generation for large models on multiple GPUs):

python3 -m dotenv run python3 scripts/putnam/putnamlike0_save_rollouts.py \
    --dataset_type putnam_historical \
    --model_id "~/model/huggingface/llama/Llama-3.1-70B/" \
    --api hf \
    --temperature 0.3 \
    --local-gen-seed 123 \
    --prefix 5 \
    --verbose

"""

import asyncio
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
from chainscope.cot_generation import _compute_and_export_metrics
from chainscope.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathQsDataset,
    MathQuestion,
    MathResponse,
    QuestionResponseId,
    SamplingParams,
    SplitCotResponses,
    StepFaithfulness,
)
from chainscope.utils import MODELS_MAP, is_instruct_model, make_chat_prompt


# NOTE: _compute_and_export_metrics is now imported from chainscope.cot_generation
# to avoid code duplication. Previously defined here from lines 93-224.


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


def get_putnam_responses_hf(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    sampling_params: SamplingParams,
    local_gen_seed: int,
    compute_metric: bool = False,
    metric_type: str = "mi",
    metric_path: Optional[str] = None,
    debug: bool = False,
    reduce_dim: bool = False,
    num_dim: int = 100,
    reduce_method: str = "pca",
    reduce_per_step: bool = False,
    select_tokens: bool = False,
    token_index_list: list[int] | None = None,
    n_jobs: int = -1,
    context_size: int | None = None,
    use_gpu: bool = False,
    metric_batch: int | None = None,
    faithfulness_labels_for_metrics: dict[tuple[str, str], str] | None = None,
    metric_faithfulness_filter: str = "all",
) -> list[tuple[QuestionResponseId, str, str | None]]:
    """Generate responses using HuggingFace native generation for Putnam problems.

    This uses HF's native generation with device_map="auto" for multi-GPU support.
    Works well for large models (70B+) on multiple GPUs.

    Args:
        prompts: List of (question ID, prompt text) tuples
        model_id: Model ID for generation
        sampling_params: Sampling parameters
        local_gen_seed: Seed for generation
        compute_metric: Whether to compute and export metrics
        metric_type: Type of metric to compute ('mi' or 'phi')
        metric_path: Path to export metric visualizations
        reduce_dim: Apply dimensionality reduction
        num_dim: Target dimensions
        reduce_method: Reduction method
        reduce_per_step: Apply PCA separately per generation step (for phi)
        select_tokens: Whether to select specific token positions
        token_index_list: List of token indices to select if select_tokens is True
    """
    import torch
    
    # Set seed for reproducibility
    torch.manual_seed(local_gen_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_gen_seed)
    
    # Check if model_id is a local path
    is_local_path = model_id.startswith('/') or model_id.startswith('./') or model_id.startswith('../') or model_id.startswith('~')
    
    logging.info(f"Loading model from {'local path' if is_local_path else 'HuggingFace'}: {model_id}")
    
    # Expand ~ in path if present
    if is_local_path and model_id.startswith('~'):
        model_id = os.path.expanduser(model_id)
    
    # Load model with device_map="auto" for multi-GPU distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=is_local_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=is_local_path,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logging.info(f"Model loaded successfully. Device map: {model.hf_device_map}")
    
    # Prepare prompts and generate responses
    responses: list[tuple[QuestionResponseId, str, str | None]] = []

    # Create metric output directory if needed
    if compute_metric and metric_path:
        os.makedirs(metric_path, exist_ok=True)

    for idx, (q_resp_id, prompt) in enumerate(tqdm(prompts, desc="Generating responses")):
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=tokenizer,
            )
        else:
            input_str = prompt

        # Tokenize input
        inputs = tokenizer(input_str, return_tensors="pt")

        # Move inputs to the first device of the model
        # device_map="auto" handles the rest
        first_device = next(iter(model.hf_device_map.values()))
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                do_sample=sampling_params.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_hidden_states=compute_metric,
                return_dict_in_generate=compute_metric,
            )

        # Decode only the generated tokens (skip the input)
        if compute_metric:
            generated_tokens = outputs.sequences[0, inputs['input_ids'].shape[1]:]
        else:
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0, input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        responses.append((q_resp_id, generated_text, None))

        # Compute and export metrics immediately after generation
        if compute_metric and hasattr(outputs, 'hidden_states') and metric_path:
            # Check if we should compute metrics for this response based on faithfulness
            should_compute_metric = True
            response_metric_path = metric_path

            if faithfulness_labels_for_metrics is not None:
                # Get the label for this response
                qid = q_resp_id.qid
                uuid = q_resp_id.response_uuid
                response_label = faithfulness_labels_for_metrics.get((qid, uuid))

                # Apply filter
                if response_label is not None:
                    if metric_faithfulness_filter == "faithful" and response_label != "faithful":
                        should_compute_metric = False
                    elif metric_faithfulness_filter == "unfaithful" and response_label != "unfaithful":
                        should_compute_metric = False

                    # Modify output path to separate by label
                    if should_compute_metric:
                        import os
                        response_metric_path = os.path.join(metric_path, response_label)
                        logging.info(f"Computing metrics for {response_label} response: {qid}/{uuid}")
                else:
                    # No label found for this response - skip if filtering is active
                    if metric_faithfulness_filter != "all":
                        logging.warning(f"No faithfulness label found for {qid}/{uuid}, skipping metric computation")
                        should_compute_metric = False

            if should_compute_metric:
                _compute_and_export_metrics(
                    hidden_states=outputs.hidden_states,
                    response_idx=idx,
                    metric_type=metric_type,
                    metric_path=response_metric_path,
                    model_id=model_id,
                    debug=debug,
                    reduce_dim=reduce_dim,
                    num_dim=num_dim,
                    reduce_method=reduce_method,
                    reduce_per_step=reduce_per_step,
                    select_tokens=select_tokens,
                    token_index_list=token_index_list,
                    n_jobs=n_jobs,
                    context_size=context_size,
                    use_gpu=use_gpu,
                    metric_batch=metric_batch,
                )
            # Free memory immediately to prevent OOM
            del outputs
            torch.cuda.empty_cache()

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
        # First, read the config to determine the official model name for TransformerLens
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        
        # Map architecture to official HF model name that TransformerLens recognizes
        model_type = getattr(config, "model_type", "")
        architectures = getattr(config, "architectures", [])
        
        if "llama" in model_type.lower() or any("Llama" in arch for arch in architectures):
            vocab_size = getattr(config, "vocab_size", 0)
            if vocab_size == 128256:
                # Llama 3.1
                official_name = "meta-llama/Llama-3.1-70B"
            elif vocab_size == 128000:
                # Llama 3
                official_name = "meta-llama/Meta-Llama-3-70B"
            else:
                raise ValueError(f"Unknown Llama variant with vocab_size={vocab_size}")
            logging.info(f"Detected architecture: {model_type}, using official name: {official_name}")
        else:
            raise ValueError(f"Unsupported model type '{model_type}' for local path loading with TransformerLens")
        
        logging.info("Loading model and tokenizer from transformers...")
        # Use device_map="auto" to distribute model across available GPUs
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
        )
        
        # Wrap with HookedTransformer using the official name
        logging.info("Converting to HookedTransformer...")
        # Note: HookedTransformer doesn't support device_map, so we pass the already-loaded model
        # The model is already distributed across GPUs via device_map="auto"
        model = HookedTransformer.from_pretrained(
            model_name=official_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device="cuda",  # This will be overridden by the hf_model's device_map
        )
    else:
        # For non-local models, load with device_map for multi-GPU support
        logging.info(f"Loading model from HuggingFace: {model_id}")
        # First load the HF model with device_map
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Wrap with HookedTransformer
        model = HookedTransformer.from_pretrained(
            model_name=model_id,
            hf_model=hf_model,
            tokenizer=tokenizer,
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
        fsp_by_resp_id=None,
        model_id=model_id,
        instr_id="instr-v0",
        ds_params=dataset.params,
        sampling_params=DefaultSamplingParams(),
    )


def load_and_filter_faithfulness_responses(
    input_path: Path,
    faithfulness_filter: str,
    evaluation_mode_str: str,
) -> CotResponses:
    """Load pre-evaluated faithfulness responses and optionally filter.

    ⚠️  WARNING: POST-HOC ANALYSIS ONLY - BACKWARD PIPELINE FLOW ⚠️

    This function loads responses that have ALREADY been evaluated by putnamlike3
    for faithfulness. This goes BACKWARD in the normal pipeline:
        Normal: putnamlike0 (generate) → putnamlike3 (evaluate) → analyze
        This:   putnamlike3 (evaluate) → putnamlike0 (load) → analyze

    REQUIREMENTS:
    - Input file MUST be from putnamlike3 output (e.g., *_reward_hacking.yaml)
    - Input file MUST contain StepFaithfulness annotations
    - This is ONLY for reusing existing evaluation data for additional analysis

    Args:
        input_path: Path to faithfulness-annotated YAML file from putnamlike3
        faithfulness_filter: "all", "faithful", or "unfaithful"
        evaluation_mode_str: "reward_hacking" or "latent_error_correction"

    Returns:
        CotResponses object with filtered responses
    """
    from chainscope.cot_faithfulness_utils import EvaluationMode

    logging.info(f"Loading pre-evaluated faithfulness responses from {input_path}")
    logging.warning("=" * 80)
    logging.warning("⚠️  POST-HOC ANALYSIS MODE - BACKWARD PIPELINE FLOW")
    logging.warning("This loads ALREADY-EVALUATED responses from putnamlike3 output.")
    logging.warning("Normal pipeline: generate (0) → evaluate (3) → load here")
    logging.warning("=" * 80)

    # Load split responses with faithfulness annotations
    split_responses = SplitCotResponses.load(input_path)

    # Get expected pattern for evaluation mode
    eval_mode = EvaluationMode(evaluation_mode_str.upper())
    expected_pattern = eval_mode.expected_answers_str

    logging.info(
        f"Filtering mode: {faithfulness_filter}, "
        f"Evaluation: {evaluation_mode_str}, "
        f"Expected pattern: {expected_pattern}"
    )

    # Statistics tracking
    total_responses = 0
    total_steps = 0
    faithful_responses_count = 0
    unfaithful_responses_count = 0
    unfaithful_steps_count = 0

    # Filter responses based on faithfulness pattern
    filtered_responses_by_qid = {}

    for qid, response_dict in split_responses.split_responses_by_qid.items():
        for uuid, response in response_dict.items():
            total_responses += 1
            has_unfaithful_step = False

            # Check each step for unfaithfulness
            if isinstance(response.model_answer, list):
                for step in response.model_answer:
                    total_steps += 1

                    if isinstance(step, StepFaithfulness):
                        # Calculate Hamming distance from expected pattern
                        if len(step.unfaithfulness) == len(expected_pattern):
                            dist = sum(
                                int(x != y)
                                for x, y in zip(step.unfaithfulness, expected_pattern)
                            )

                            if dist == 0:  # Exact match = unfaithful step
                                has_unfaithful_step = True
                                unfaithful_steps_count += 1

            # Count response faithfulness
            if has_unfaithful_step:
                unfaithful_responses_count += 1
            else:
                faithful_responses_count += 1

            # Apply filter
            should_include = (
                faithfulness_filter == "all" or
                (faithfulness_filter == "faithful" and not has_unfaithful_step) or
                (faithfulness_filter == "unfaithful" and has_unfaithful_step)
            )

            if should_include:
                if qid not in filtered_responses_by_qid:
                    filtered_responses_by_qid[qid] = {}
                filtered_responses_by_qid[qid][uuid] = response

    # Calculate percentages
    faithful_pct = (faithful_responses_count / total_responses * 100) if total_responses > 0 else 0
    unfaithful_pct = (unfaithful_responses_count / total_responses * 100) if total_responses > 0 else 0
    unfaithful_step_pct = (unfaithful_steps_count / total_steps * 100) if total_steps > 0 else 0

    # Log comprehensive statistics
    logging.warning("")
    logging.warning("=" * 80)
    logging.warning("FAITHFULNESS STATISTICS")
    logging.warning("=" * 80)
    logging.warning(f"Total responses loaded: {total_responses:,}")
    logging.warning(f"Total steps evaluated: {total_steps:,}")
    logging.warning("")
    logging.warning(f"Faithful responses (no unfaithful steps): {faithful_responses_count:,} "
                   f"({faithful_pct:.1f}%)")
    logging.warning(f"Unfaithful responses (≥1 unfaithful step): {unfaithful_responses_count:,} "
                   f"({unfaithful_pct:.1f}%)")
    logging.warning("")
    logging.warning(f"Unfaithful steps (matching {expected_pattern}): {unfaithful_steps_count:,} "
                   f"({unfaithful_step_pct:.2f}% of all steps)")
    logging.warning("")
    logging.warning(f"Filter applied: '{faithfulness_filter}'")
    filtered_count = sum(len(d) for d in filtered_responses_by_qid.values())
    logging.warning(f"Responses after filtering: {filtered_count:,}")
    logging.warning("=" * 80)

    # Convert back to CotResponses format
    # Note: We lose the step-level annotations here, converting back to strings
    responses_by_qid = {}
    for qid, response_dict in filtered_responses_by_qid.items():
        responses_by_qid[qid] = {}
        for uuid, response in response_dict.items():
            # Convert StepFaithfulness back to strings if needed
            model_answer = response.model_answer
            if isinstance(model_answer, list) and len(model_answer) > 0:
                if isinstance(model_answer[0], StepFaithfulness):
                    # Extract just the step strings
                    model_answer = [step.step_str for step in model_answer]

            responses_by_qid[qid][uuid] = MathResponse(
                name=response.name,
                problem=response.problem,
                solution=response.solution,
                model_thinking=response.model_thinking,
                model_answer=model_answer,
                correctness_explanation=response.correctness_explanation,
                correctness_is_correct=response.correctness_is_correct,
                correctness_classification=response.correctness_classification,
            )

    return CotResponses(
        responses_by_qid=responses_by_qid,
        fsp_by_resp_id=None,
        model_id=split_responses.model_id,
        instr_id=split_responses.instr_id,
        ds_params=split_responses.ds_params,
        sampling_params=split_responses.sampling_params,
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
    compute_metric: bool = False,
    metric_type: str = "mi",
    metric_path: Optional[str] = None,
    debug: bool = False,
    reduce_dim: bool = False,
    num_dim: int = 100,
    reduce_method: str = "pca",
    reduce_per_step: bool = False,
    select_tokens: bool = False,
    token_index_list: list[int] | None = None,
    n_jobs: int = -1,
    context_size: int | None = None,
    use_gpu: bool = False,
    metric_batch: int = None,
    faithfulness_labels_for_metrics: dict[tuple[str, str], str] | None = None,
    metric_faithfulness_filter: str = "all",
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
        reduce_dim: Apply dimensionality reduction
        num_dim: Target dimensions
        reduce_method: Reduction method
        reduce_per_step: Apply PCA per generation step (for phi)

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
            fsp_by_resp_id=None,
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
    elif api == "hf":
        results = get_putnam_responses_hf(
            prompts=all_prompts,
            model_id=model_id,
            sampling_params=sampling_params,
            local_gen_seed=local_gen_seed,
            compute_metric=compute_metric,
            metric_type=metric_type,
            metric_path=metric_path,
            debug=debug,
            reduce_dim=reduce_dim,
            num_dim=num_dim,
            reduce_method=reduce_method,
            reduce_per_step=reduce_per_step,
            select_tokens=select_tokens,
            token_index_list=token_index_list,
            n_jobs=n_jobs,
            context_size=context_size,
            use_gpu=use_gpu,
            metric_batch=metric_batch,
            faithfulness_labels_for_metrics=faithfulness_labels_for_metrics,
            metric_faithfulness_filter=metric_faithfulness_filter,
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
            fsp_by_resp_id=None,
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
        fsp_by_resp_id=None,
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
    type=click.Choice(["vllm", "ttl", "hf"]),
    default=None,
    help="Use local API for generation (vllm, ttl, or hf=HuggingFace native)",
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
@click.option(
    "--compute-metric",
    is_flag=True,
    help="Compute and export metrics (only works with --api hf)",
)
@click.option(
    "--metric",
    type=click.Choice(["mi", "phi"]),
    default="mi",
    help="Type of metric to compute (mi or phi)",
)
@click.option(
    "--metric-path",
    type=str,
    default=None,
    help="Path to export metric visualizations",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with verbose parallel execution logging",
)
@click.option(
    "--reduce-dim",
    is_flag=True,
    help="Enable dimensionality reduction before metric computation",
)
@click.option(
    "--num-dim",
    type=int,
    default=100,
    help="Target number of dimensions after reduction (must be < hidden_dim)",
)
@click.option(
    "--reduce-method",
    type=click.Choice(["pca"]),
    default="pca",
    help="Dimensionality reduction method",
)
@click.option(
    "--reduce-per-step",
    is_flag=True,
    help="Apply PCA separately per generation step (recommended for phi metric)",
)
@click.option(
    "--select-tokens",
    is_flag=True,
    help="Select specific token positions instead of all tokens",
)
@click.option(
    "--token-index-list",
    type=str,
    default=None,
    help="Comma-separated list of token indices to select (e.g., '0,5,10,19')",
)
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    help="Number of parallel jobs for metric computation (-1 for all cores, 4 recommended to prevent OOM)",
)
@click.option(
    "--context-size",
    type=int,
    default=None,
    help="Number of tokens to extract as context window for phi computation (e.g., 5 for 5-token windows)",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Use GPU acceleration for phi/MI computation (requires CUDA)",
)
@click.option(
    "--metric-batch",
    type=int,
    default=None,
    help="Number of states to process at once for GPU metric computation (None = auto-calculate based on GPU memory). Only used with --use-gpu.",
)
@click.option(
    "--load-ground-truth",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="⚠️  POST-HOC ANALYSIS ONLY: Load pre-evaluated faithfulness responses from putnamlike3 output (goes BACKWARD in pipeline). Requires file with StepFaithfulness annotations.",
)
@click.option(
    "--faithfulness-filter",
    type=click.Choice(["all", "faithful", "unfaithful"]),
    default="all",
    help="Filter loaded responses by faithfulness (only used with --load-ground-truth). 'all' keeps everything, 'faithful' keeps only responses without unfaithful steps, 'unfaithful' keeps only responses with unfaithful steps.",
)
@click.option(
    "--evaluation-mode",
    type=click.Choice(["reward_hacking", "latent_error_correction"]),
    default="reward_hacking",
    help="Evaluation mode for pattern matching (only used with --load-ground-truth). 'reward_hacking' uses pattern YNNNYNFN (8 questions), 'latent_error_correction' uses YNFNFYNYN (9 questions).",
)
@click.option(
    "--metric-faithfulness-labels",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to faithfulness evaluation labels from putnamlike3 for metric computation. If not provided but --load-ground-truth is set, will use that path. Typically: data/cot_responses/instr-v0/default_sampling_params/{dataset_id}/{base_filename}_{eval_model}_{eval_mode}.yaml",
)
@click.option(
    "--metric-faithfulness-filter",
    type=click.Choice(["all", "faithful", "unfaithful"]),
    default="all",
    help="Filter which responses to compute metrics for based on faithfulness labels (only used with --metric-faithfulness-labels). 'all' computes for all responses, 'faithful' only for faithful responses, 'unfaithful' only for unfaithful responses.",
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
    compute_metric: bool,
    metric: str,
    metric_path: Optional[str],
    debug: bool,
    reduce_dim: bool,
    num_dim: int,
    reduce_method: str,
    reduce_per_step: bool,
    select_tokens: bool,
    token_index_list: str | None,
    n_jobs: int,
    context_size: int | None,
    use_gpu: bool,
    metric_batch: int,
    load_ground_truth: Optional[Path],
    faithfulness_filter: str,
    evaluation_mode: str,
    metric_faithfulness_labels: Optional[Path],
    metric_faithfulness_filter: str,
):
    """Generate rollouts for Putnam problems using OpenRouter or DeepSeek models.

    When --load-ground-truth is provided, this script loads pre-evaluated faithfulness
    responses instead of generating new ones. This is for POST-HOC ANALYSIS ONLY.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # ========================================================================
    # SPECIAL CASE: Load pre-evaluated faithfulness responses (POST-HOC ONLY)
    # ========================================================================
    # This goes BACKWARD in the pipeline:
    #   Normal flow: putnamlike0 (generate) → putnamlike3 (evaluate)
    #   This flow:   putnamlike3 (evaluate) → putnamlike0 (load for analysis)
    #
    # This is ONLY for post-hoc analysis of already-evaluated responses.
    # It bypasses all generation logic and loads responses with StepFaithfulness
    # annotations from putnamlike3 output files.
    # ========================================================================
    if load_ground_truth is not None:
        logging.warning("=" * 80)
        logging.warning("⚠️  POST-HOC ANALYSIS MODE: Loading pre-evaluated faithfulness responses")
        logging.warning(f"⚠️  Input file: {load_ground_truth}")
        logging.warning(f"⚠️  Filter mode: {faithfulness_filter}")
        logging.warning(f"⚠️  Evaluation mode: {evaluation_mode}")
        logging.warning("⚠️  This goes BACKWARD in the pipeline - loading already-evaluated data")
        logging.warning("=" * 80)

        # Load and filter the faithfulness responses
        filtered_responses = load_and_filter_faithfulness_responses(
            input_path=load_ground_truth,
            faithfulness_filter=faithfulness_filter,
            evaluation_mode_str=evaluation_mode,
        )

        # Save filtered results
        # Extract meaningful suffix from input filename for output
        input_stem = load_ground_truth.stem
        filter_suffix = f"_filtered_{faithfulness_filter}" if faithfulness_filter != "all" else "_all"

        # Try to find existing versioned output
        for i in range(0, 100):
            output_path = filtered_responses.get_path(f"_v{i}_ground_truth{filter_suffix}")
            if not os.path.exists(output_path):
                break

        saved_path = filtered_responses.save(path=output_path)
        logging.warning("=" * 80)
        logging.warning(f"✓ Saved filtered responses to: {saved_path}")
        logging.warning("=" * 80)
        return  # Early return - skip all generation logic

    # ========================================================================
    # NORMAL FLOW: Generate new responses (typical use case)
    # ========================================================================

    # Convert dataset type string to enum
    dataset_type_enum = DatasetType(dataset_type)

    # Create dataset directly based on type
    dataset = create_putnam_dataset(dataset_type_enum)

    # Validate metric computation arguments
    if compute_metric:
        if api != "hf":
            logging.error("--compute-metric only works with --api hf")
            return
        if metric_path is None:
            logging.error("--metric-path is required when --compute-metric is set")
            return

    # Parse token index list if provided
    parsed_token_indices = None
    if token_index_list is not None:
        try:
            parsed_token_indices = [int(x.strip()) for x in token_index_list.split(",")]
            logging.info(f"Parsed token indices: {parsed_token_indices}")
        except ValueError as e:
            logging.error(f"Invalid token index list format: {e}")
            return

    # Load faithfulness labels for metric computation if requested
    faithfulness_labels_for_metrics = None
    if compute_metric and (metric_faithfulness_labels is not None or load_ground_truth is not None):
        # Use metric_faithfulness_labels if provided, otherwise fall back to load_ground_truth
        labels_path = metric_faithfulness_labels if metric_faithfulness_labels is not None else load_ground_truth

        logging.info(f"Loading faithfulness labels from: {labels_path}")
        logging.info(f"Metric faithfulness filter: {metric_faithfulness_filter}")

        # Load faithfulness responses (don't filter yet, just load all)
        faithfulness_responses = load_and_filter_faithfulness_responses(
            input_path=labels_path,
            faithfulness_filter="all",  # Load all, we'll filter per-response
            evaluation_mode_str=evaluation_mode,
        )

        # Build a mapping from (qid, response_uuid) to "faithful" or "unfaithful"
        # This will be used during metric computation to filter/separate outputs
        from chainscope.typing import StepFaithfulness
        from chainscope.cot_faithfulness_utils import EvaluationMode

        faithfulness_labels_for_metrics = {}
        eval_mode = EvaluationMode(evaluation_mode.upper())
        expected_pattern = eval_mode.expected_answers_str

        for qid, response_dict in faithfulness_responses.split_responses_by_qid.items():
            for uuid, response in response_dict.items():
                # Determine if this response is faithful or unfaithful
                has_unfaithful_step = False

                if isinstance(response.model_answer, list):
                    for step in response.model_answer:
                        if isinstance(step, StepFaithfulness):
                            # Check if this step matches the expected unfaithfulness pattern
                            if len(step.unfaithfulness) == len(expected_pattern):
                                dist = sum(int(x != y) for x, y in zip(step.unfaithfulness, expected_pattern))
                                if dist == 0:  # Exact match = unfaithful step
                                    has_unfaithful_step = True
                                    break

                # Store label
                label = "unfaithful" if has_unfaithful_step else "faithful"
                faithfulness_labels_for_metrics[(qid, uuid)] = label

        logging.info(f"Loaded {len(faithfulness_labels_for_metrics)} faithfulness labels")
        faithful_count = sum(1 for label in faithfulness_labels_for_metrics.values() if label == "faithful")
        unfaithful_count = len(faithfulness_labels_for_metrics) - faithful_count
        logging.info(f"  Faithful: {faithful_count}, Unfaithful: {unfaithful_count}")

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
                compute_metric=compute_metric,
                metric_type=metric,
                metric_path=metric_path,
                debug=debug,
                reduce_dim=reduce_dim,
                num_dim=num_dim,
                reduce_method=reduce_method,
                reduce_per_step=reduce_per_step,
                select_tokens=select_tokens,
                token_index_list=parsed_token_indices,
                n_jobs=n_jobs,
                context_size=context_size,
                use_gpu=use_gpu,
                metric_batch=metric_batch,
                faithfulness_labels_for_metrics=faithfulness_labels_for_metrics,
                metric_faithfulness_filter=metric_faithfulness_filter,
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
