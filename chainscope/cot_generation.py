import logging
import os
import random
from pathlib import Path
from uuid import uuid4

import torch as t
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from vllm import LLM
from vllm import SamplingParams as VLLMSamplingParams

from chainscope.questions import QsDataset
from chainscope.typing import *
from chainscope.utils import is_instruct_model, make_chat_prompt


def _compute_and_export_metrics(
    hidden_states: tuple,
    response_idx: int,
    metric_type: str,
    metric_path: str,
    model_id: str,
    export_json: bool = False,
    debug: bool = False,
    reduce_dim: bool = False,
    num_dim: int = 100,
    reduce_method: str = "pca",
    reduce_per_step: bool = False,
    select_tokens: bool = False,
    token_index_list: list[int] | None = None,
):
    """Compute and export metrics from hidden states of a single response.

    Args:
        hidden_states: Hidden states from model generation for a single response
        response_idx: Index of the response for file naming
        metric_type: Type of metric to compute ('mi' or 'phi')
        metric_path: Path to export metric visualizations
        model_id: Model ID for title generation
        export_json: Whether to export JSON alongside PNG for multi-token responses
        debug: Enable debug mode with verbose parallel execution logging
        reduce_dim: Whether to apply dimensionality reduction
        num_dim: Target dimensions for reduction
        reduce_method: Method for reduction ('pca')
        reduce_per_step: Apply PCA separately per generation step (for phi)
        select_tokens: Whether to select specific token positions
        token_index_list: List of token indices to select if select_tokens is True
    """
    import json

    try:
        from src import compute_metric_matrix
        from eval import save_matrix_viz
    except ImportError:
        logging.error("LINE package not installed. Install with: uv pip install -e ../LINE")
        return

    # Create output directory if it doesn't exist
    os.makedirs(metric_path, exist_ok=True)

    try:
        # Extract last layer states from each generation step
        # hidden_states is a tuple of (num_generated_tokens,) where each element
        # is a tuple of (num_layers,) tensors with shape (batch_size, seq_len, hidden_dim)
        # Note: seq_len grows with each generation step (contains full sequence up to that point)
        last_layer_states = [step[-1] for step in hidden_states]

        # Convert to float32 if needed (scikit-learn doesn't support bfloat16)
        last_layer_states = [state.float() if state.dtype == t.bfloat16 else state
                             for state in last_layer_states]

        logging.info(f"Response {response_idx}: {len(last_layer_states)} generation steps, "
                    f"each with shape {last_layer_states[0].shape}")

        # Extract only the last token from each step to normalize format
        # This converts each tensor from (batch, seq_len, hidden) to (batch, 1, hidden)
        # where the extracted token represents the newly generated token at that step
        last_layer_states = [state[:, -1:, :] for state in last_layer_states]

        logging.info(f"Response {response_idx}: After extracting last token from each step, "
                    f"shape: {last_layer_states[0].shape}")

        # Handle single-token responses (export single-valued metric as JSON)
        if len(last_layer_states) < 2:
            logging.info(f"Response {response_idx}: Single-token response detected. "
                        f"Computing single-valued {metric_type.upper()} metric.")

            # Compute single-valued metric from the single hidden state
            # For MI: compute self-information (entropy of the state)
            # For Phi: use the same approach
            single_state = last_layer_states[0]

            # Compute metric using the same framework but with single state
            if metric_type == "phi":
                # Split at token level: split_index should be seq_len // 2
                _, seq_len, _ = single_state.shape
                split_index = seq_len // 2
                metric_value = compute_metric_matrix(
                    [single_state],
                    metric=metric_type,
                    method='knn',
                    split_index=split_index,
                    debug=debug,
                    reduce_dim=reduce_dim,
                    num_dim=num_dim,
                    reduce_method=reduce_method,
                    reduce_per_step=reduce_per_step,
                    select_tokens=select_tokens,
                    token_index_list=token_index_list,
                )[0, 0]  # Extract scalar from 1x1 matrix
            else:
                metric_value = compute_metric_matrix(
                    [single_state],
                    metric=metric_type,
                    method='knn',
                    debug=debug,
                    reduce_dim=reduce_dim,
                    num_dim=num_dim,
                    reduce_method=reduce_method,
                    reduce_per_step=reduce_per_step,
                    select_tokens=select_tokens,
                    token_index_list=token_index_list,
                )[0, 0]  # Extract scalar from 1x1 matrix

            # Export as JSON
            metric_data = {
                "metric_type": metric_type,
                "metric_value": float(metric_value),
            }

            output_file = os.path.join(metric_path, f'response_{response_idx}_{metric_type}_single.json')
            with open(output_file, 'w') as f:
                json.dump(metric_data, f, indent=2)
            logging.info(f"Saved {metric_type.upper()} single-valued metric to {output_file}")
            return

        # Compute metric matrix for multi-token responses
        if metric_type == "phi":
            # For phi, we need to specify split_index at token level
            _, seq_len, _ = last_layer_states[0].shape
            split_index = seq_len // 2
            metric_matrix = compute_metric_matrix(
                last_layer_states,
                metric=metric_type,
                method='knn',
                split_index=split_index,
                debug=debug,
                reduce_dim=reduce_dim,
                num_dim=num_dim,
                reduce_method=reduce_method,
                reduce_per_step=reduce_per_step,
                select_tokens=select_tokens,
                token_index_list=token_index_list,
            )
        else:
            metric_matrix = compute_metric_matrix(
                last_layer_states,
                metric=metric_type,
                method='knn',
                debug=debug,
                reduce_dim=reduce_dim,
                num_dim=num_dim,
                reduce_method=reduce_method,
                reduce_per_step=reduce_per_step,
                select_tokens=select_tokens,
                token_index_list=token_index_list,
            )

        # Save visualization (PNG)
        model_name = model_id.split("/")[-1]
        output_file = os.path.join(metric_path, f'response_{response_idx}_{metric_type}.png')
        save_matrix_viz(
            metric_matrix,
            file_path=output_file,
            title=f'{model_name} - {metric_type.upper()} Matrix (Response {response_idx})',
            metric_type=metric_type,
            figsize=(10, 8),
            dpi=300,
            show_annotations=False,
        )
        logging.info(f"Saved {metric_type.upper()} visualization to {output_file}")

        # Optionally export JSON alongside PNG for multi-token responses
        if export_json:
            metric_data = {
                "metric_type": metric_type,
                "metric_matrix": metric_matrix.tolist(),
            }
            json_file = os.path.join(metric_path, f'response_{response_idx}_{metric_type}.json')
            with open(json_file, 'w') as f:
                json.dump(metric_data, f, indent=2)
            logging.info(f"Saved {metric_type.upper()} matrix data to {json_file}")

    except Exception as e:
        logging.error(f"Error computing metrics for response {response_idx}: {e}")


def build_fsp_prompt(
    model_id_for_fsp: str,
    fsp_size: int,
    instr_id: str,
    ds_params: DatasetParams,
    sampling_params: SamplingParams,
    fsp_seed: int,
    instruction_cache: dict[str, Instructions],
    cot_responses_cache: dict[str, CotResponses],
    qs_dataset_cache: dict[str, QsDataset],
) -> str:
    random.seed(fsp_seed)

    # Get Instructions from cache or load them
    if instr_id in instruction_cache:
        instructions = instruction_cache[instr_id]
    else:
        instructions = Instructions.load(instr_id)
        instruction_cache[instr_id] = instructions

    # Load CoT responses from model_id_for_fsp for this dataset
    cot_responses_path = ds_params.cot_responses_path(
        instr_id=instr_id,
        model_id=model_id_for_fsp,
        sampling_params=sampling_params,
    )

    # Convert Path to string for dictionary key
    cot_responses_path_str = str(cot_responses_path)
    if cot_responses_path_str in cot_responses_cache:
        cot_responses = cot_responses_cache[cot_responses_path_str]
    else:
        cot_responses = CotResponses.load(cot_responses_path)
        cot_responses_cache[cot_responses_path_str] = cot_responses

    qs_dataset_path = ds_params.qs_dataset_path

    # Convert Path to string for dictionary key
    qs_dataset_path_str = str(qs_dataset_path)
    if qs_dataset_path_str in qs_dataset_cache:
        qs_dataset = qs_dataset_cache[qs_dataset_path_str]
    else:
        qs_dataset = QsDataset.load_from_path(qs_dataset_path)
        qs_dataset_cache[qs_dataset_path_str] = qs_dataset

    cot_prompts = []
    for qid, responses in cot_responses.responses_by_qid.items():
        q_str = qs_dataset.question_by_qid[qid].q_str
        prompt = instructions.cot.format(question=q_str)
        for resp in responses.values():
            assert isinstance(resp, str)
            prompt_and_resp = f"{prompt}{resp}"
            cot_prompts.append(prompt_and_resp)

    # Choose fsp_size random prompts
    fsp_prompts = random.sample(cot_prompts, fsp_size)
    fsp_prompt = "\n\n".join(fsp_prompts)

    return fsp_prompt


def get_local_responses_vllm(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params_list: list[DatasetParams],
    sampling_params: SamplingParams,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    qid_to_dataset: dict[str, str],
    batch_size: int = 4096,
) -> list[tuple[QuestionResponseId, str, str | None]]:
    assert instr_id == "instr-wm", "Only instr-wm is supported for local generation"
    if model_id_for_fsp is not None:
        assert not is_instruct_model(model_id), "Why?"

    # Initialize caches
    instruction_cache: dict[str, Instructions] = {}
    cot_responses_cache: dict[str, CotResponses] = {}
    qs_dataset_cache: dict[str, QsDataset] = {}

    # Initialize vLLM engine
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=t.cuda.device_count(),
    )

    instr_prefix = "Here is a question with a clear YES or NO answer"

    # Convert our sampling params to vLLM format
    vllm_params = VLLMSamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=sampling_params.max_new_tokens,
        stop=["**NO**", "**YES**", "\n\nNO", "\n\nYES", instr_prefix],
        include_stop_str_in_output=True,
    )

    # Prepare prompts
    prompt_texts = []
    q_resp_ids = []
    fsp_for_output: list[str | None] = []
    for q_resp_id, prompt in tqdm(prompts, desc="Preparing prompts"):
        current_fsp_prompt: str | None = None
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=llm.get_tokenizer(),  # type: ignore
            )
        else:
            # Get FSP prompt for this dataset if needed
            if model_id_for_fsp is not None:
                dataset_id = qid_to_dataset[q_resp_id.qid]
                ds_idx = next(
                    i for i, ds in enumerate(ds_params_list) if ds.id == dataset_id
                )
                ds_params = ds_params_list[ds_idx]
                fsp_prompt = build_fsp_prompt(
                    model_id_for_fsp=model_id_for_fsp,
                    fsp_size=fsp_size,
                    instr_id=instr_id,
                    ds_params=ds_params,
                    sampling_params=sampling_params,
                    fsp_seed=fsp_seed,
                    instruction_cache=instruction_cache,
                    cot_responses_cache=cot_responses_cache,
                    qs_dataset_cache=qs_dataset_cache,
                )
                input_str = f"{fsp_prompt}\n\n{prompt}"
                current_fsp_prompt = fsp_prompt
            else:
                input_str = prompt

        prompt_texts.append(input_str)
        q_resp_ids.append(q_resp_id)
        fsp_for_output.append(current_fsp_prompt)

    # Generate responses using vLLM in batches
    logging.info(
        f"Generating {len(prompt_texts)} responses with batch size {batch_size}"
    )
    all_outputs = []
    for i in tqdm(range(0, len(prompt_texts), batch_size), desc="Processing batches"):
        batch_prompts = prompt_texts[i : i + batch_size]
        batch_outputs = llm.generate(batch_prompts, vllm_params, use_tqdm=True)
        all_outputs.extend(batch_outputs)
    logging.info(f"Generated {len(all_outputs)} responses")

    # Format responses
    responses: list[tuple[QuestionResponseId, str, str | None]] = []
    for q_resp_id, output, fsp in tqdm(
        zip(q_resp_ids, all_outputs, fsp_for_output), desc="Processing responses", total=len(q_resp_ids)
    ):
        generated_text = output.outputs[0].text

        if instr_prefix in generated_text:
            generated_text = generated_text.replace(instr_prefix, "")

        responses.append((q_resp_id, generated_text, fsp))

    return responses


def get_local_responses_hf(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params_list: list[DatasetParams],
    sampling_params: SamplingParams,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    local_gen_seed: int,
    qid_to_dataset: dict[str, str],
    compute_metric: bool = False,
    metric_type: str = "mi",
    metric_path: str | None = None,
    debug: bool = False,
    reduce_dim: bool = False,
    num_dim: int = 100,
    reduce_method: str = "pca",
    reduce_per_step: bool = False,
    select_tokens: bool = False,
    token_index_list: list[int] | None = None,
) -> list[tuple[QuestionResponseId, str, str | None]]:
    """Generate responses using HuggingFace native generation.

    Uses device_map="auto" for multi-GPU distribution. Works well for large models.

    Args:
        prompts: List of (question ID, prompt text) tuples
        model_id: Name of the model to use
        instr_id: Instruction ID
        ds_params_list: List of dataset parameters
        sampling_params: Sampling parameters
        model_id_for_fsp: Model ID for few-shot prompting (optional)
        fsp_size: Number of few-shot examples
        fsp_seed: Seed for few-shot example selection
        local_gen_seed: Seed for generation
        qid_to_dataset: Mapping from question IDs to dataset IDs
        compute_metric: Whether to compute and export metrics
        metric_type: Type of metric to compute ('mi' or 'phi')
        metric_path: Path to export metric visualizations
        reduce_dim: Apply dimensionality reduction
        num_dim: Target dimensions
        reduce_method: Reduction method
        reduce_per_step: Apply PCA separately per generation step (for phi)
        select_tokens: Whether to select specific token positions
        token_index_list: List of token indices to select if select_tokens is True

    Returns:
        List of (question ID, generated response, fsp_prompt or None) tuples
    """
    import os
    
    assert instr_id == "instr-wm", "Only instr-wm is supported for local generation"
    if model_id_for_fsp is not None:
        assert not is_instruct_model(model_id), "Why?"
    
    # Initialize caches
    instruction_cache: dict[str, Instructions] = {}
    cot_responses_cache: dict[str, CotResponses] = {}
    qs_dataset_cache: dict[str, QsDataset] = {}
    
    # Set seed for reproducibility
    t.manual_seed(local_gen_seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(local_gen_seed)
    
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
        torch_dtype=t.bfloat16,
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
    
    instr_prefix = "Here is a question with a clear YES or NO answer"
    stop_tokens = ["**NO**", "**YES**", "\n\nNO", "\n\nYES", instr_prefix]

    # Prepare prompts and generate responses
    responses: list[tuple[QuestionResponseId, str, str | None]] = []

    # Create metric output directory if needed
    if compute_metric and metric_path:
        os.makedirs(metric_path, exist_ok=True)

    for idx, (q_resp_id, prompt) in enumerate(tqdm(prompts, desc="Generating responses")):
        current_fsp_prompt: str | None = None
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=tokenizer,
            )
        else:
            # Get FSP prompt for this dataset if needed
            if model_id_for_fsp is not None:
                dataset_id = qid_to_dataset[q_resp_id.qid]
                ds_idx = next(
                    i for i, ds in enumerate(ds_params_list) if ds.id == dataset_id
                )
                ds_params = ds_params_list[ds_idx]
                fsp_prompt = build_fsp_prompt(
                    model_id_for_fsp=model_id_for_fsp,
                    fsp_size=fsp_size,
                    instr_id=instr_id,
                    ds_params=ds_params,
                    sampling_params=sampling_params,
                    fsp_seed=fsp_seed,
                    instruction_cache=instruction_cache,
                    cot_responses_cache=cot_responses_cache,
                    qs_dataset_cache=qs_dataset_cache,
                )
                input_str = f"{fsp_prompt}\n\n{prompt}"
                current_fsp_prompt = fsp_prompt
            else:
                input_str = prompt

        # Tokenize input
        inputs = tokenizer(input_str, return_tensors="pt")

        # Move inputs to the first device of the model
        first_device = next(iter(model.hf_device_map.values()))
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        # Generate
        with t.inference_mode():
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

        # Handle stop tokens (truncate at first occurrence)
        for stop_token in stop_tokens:
            if stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0]
                break

        # Clean up the instr_prefix if it appears
        if instr_prefix in generated_text:
            generated_text = generated_text.replace(instr_prefix, "")

        responses.append((q_resp_id, generated_text, current_fsp_prompt))

        # Compute and export metrics immediately after generation
        if compute_metric and hasattr(outputs, 'hidden_states') and metric_path:
            _compute_and_export_metrics(
                hidden_states=outputs.hidden_states,
                response_idx=idx,
                metric_type=metric_type,
                metric_path=metric_path,
                model_id=model_id,
                debug=debug,
                reduce_dim=reduce_dim,
                num_dim=num_dim,
                reduce_method=reduce_method,
                reduce_per_step=reduce_per_step,
                select_tokens=select_tokens,
                token_index_list=token_index_list,
            )
            # Free memory immediately to prevent OOM
            del outputs
            t.cuda.empty_cache()

    return responses


def get_local_responses_tl(
    prompts: list[tuple[QuestionResponseId, str]],
    model_id: str,
    instr_id: str,
    ds_params_list: list[DatasetParams],
    sampling_params: SamplingParams,
    model_id_for_fsp: str | None,
    fsp_size: int,
    fsp_seed: int,
    local_gen_seed: int,
    qid_to_dataset: dict[str, str],
) -> list[tuple[QuestionResponseId, str, str | None]]:
    """Generate responses using TransformerLens framework.

    Args:
        prompts: List of (question ID, prompt text) tuples
        model_id: Name of the model to use
        instr_id: Instruction ID
        ds_params_list: List of dataset parameters
        sampling_params: Sampling parameters
        model_id_for_fsp: Model ID for few-shot prompting (optional)
        fsp_size: Number of few-shot examples
        fsp_seed: Seed for few-shot example selection
        local_gen_seed: Seed for generation
        qid_to_dataset: Mapping from question IDs to dataset IDs

    Returns:
        List of (question ID, generated response, fsp_prompt or None) tuples
    """
    assert instr_id == "instr-wm", "Only instr-wm is supported for local generation"
    if model_id_for_fsp is not None:
        assert not is_instruct_model(model_id), "Why?"

    # Initialize caches
    instruction_cache: dict[str, Instructions] = {}
    cot_responses_cache: dict[str, CotResponses] = {}
    qs_dataset_cache: dict[str, QsDataset] = {}

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
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=t.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
        )
        
        # Wrap with HookedTransformer using the official name
        logging.info("Converting to HookedTransformer...")
        model = HookedTransformer.from_pretrained(
            model_name=official_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device="cuda",
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name=model_id,
            device="cuda",
        )
    assert model.tokenizer is not None, "Tokenizer is not initialized"

    instr_prefix = "Here is a question with a clear YES or NO answer"
    stop_tokens = ["**NO**", "**YES**", "\n\nNO", "\n\nYES", instr_prefix]

    # Prepare prompts
    responses: list[tuple[QuestionResponseId, str, str | None]] = []
    for q_resp_id, prompt in tqdm(prompts, desc="Generating responses"):
        current_fsp_prompt: str | None = None
        if is_instruct_model(model_id):
            input_str = make_chat_prompt(
                instruction=prompt,
                tokenizer=model.tokenizer,  # type: ignore
            )
        else:
            # Get FSP prompt for this dataset if needed
            if model_id_for_fsp is not None:
                dataset_id = qid_to_dataset[q_resp_id.qid]
                ds_idx = next(
                    i for i, ds in enumerate(ds_params_list) if ds.id == dataset_id
                )
                ds_params = ds_params_list[ds_idx]
                fsp_prompt = build_fsp_prompt(
                    model_id_for_fsp=model_id_for_fsp,
                    fsp_size=fsp_size,
                    instr_id=instr_id,
                    ds_params=ds_params,
                    sampling_params=sampling_params,
                    fsp_seed=fsp_seed,
                    instruction_cache=instruction_cache,
                    cot_responses_cache=cot_responses_cache,
                    qs_dataset_cache=qs_dataset_cache,
                )
                input_str = f"{fsp_prompt}\n\n{prompt}"
                current_fsp_prompt = fsp_prompt
            else:
                input_str = prompt

        # Tokenize input
        tokens = model.to_tokens(input_str, prepend_bos=True).to(model.cfg.device)
        assert isinstance(tokens, t.Tensor)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1

        # Generate the full sequence at once
        with t.inference_mode():
            generated = model.generate(
                tokens,
                max_new_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                return_type="tokens",
                verbose=False,
            )
            assert isinstance(
                generated, t.Tensor
            )  # : Int[t.Tensor, "1 pos_plus_new_tokens"]
            assert generated.ndim == 2

        # Convert output tokens to text
        generated_text = model.tokenizer.batch_decode(
            generated[:, tokens.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        assert isinstance(
            generated_text, str
        ), f"Generated text is not a string: {type(generated_text)}, {generated_text}"

        # Find the first occurrence of any stop sequence and truncate
        min_stop_idx = len(generated_text)
        for stop_seq in stop_tokens:
            stop_idx = generated_text.find(stop_seq)
            if stop_idx != -1 and stop_idx < min_stop_idx:
                min_stop_idx = stop_idx + len(stop_seq)

        # Truncate at the earliest stop sequence
        generated_text = generated_text[:min_stop_idx]

        # Clean up response
        if instr_prefix in generated_text:
            generated_text = generated_text.replace(instr_prefix, "")

        responses.append((q_resp_id, generated_text, current_fsp_prompt))

    return responses


def create_batch_of_cot_prompts(
    question_dataset: QsDataset,
    instructions: Instructions,
    question_type: Literal["yes-no", "open-ended"],
    n_responses: int,
    existing_responses: CotResponses | None = None,
) -> list[tuple[QuestionResponseId, str]]:
    """Create a batch of CoT prompts for questions that need responses.

    Args:
        question_dataset: Dataset containing questions
        instructions: Instructions for CoT generation
        question_type: Type of questions to generate responses for
        n_responses: Number of responses needed per question
        existing_responses: Existing responses to skip

    Returns:
        List of tuples containing (question response ID, prompt)
    """
    batch_items: list[tuple[QuestionResponseId, str]] = []
    for qid, q in question_dataset.question_by_qid.items():
        # Get existing responses for this question
        existing_q_responses = {}
        if (
            existing_responses is not None
            and qid in existing_responses.responses_by_qid
        ):
            existing_q_responses = existing_responses.responses_by_qid[qid]

        # Calculate how many more responses we need
        n_existing = len(existing_q_responses)
        n_needed = max(0, n_responses - n_existing)

        if n_needed == 0:
            continue

        if question_type == "yes-no":
            q_str = q.q_str
            prompt = instructions.cot.format(question=q_str)
        else:
            q_str = q.q_str_open_ended
            prompt = instructions.open_ended_cot.format(question=q_str)

        # Create n_needed items for this question
        for _ in range(n_needed):
            q_response_id = QuestionResponseId(qid=qid, uuid=str(uuid4()))
            batch_items.append((q_response_id, prompt))

    return batch_items


def create_cot_responses(
    responses_by_qid: dict[str, dict[str, MathResponse | AtCoderResponse | str]] | None,
    new_responses: list[tuple[QuestionResponseId, str, str | None]],
    model_id: str,
    instr_id: str,
    ds_params: DatasetParams,
    sampling_params: SamplingParams,
) -> CotResponses:
    """Create CotResponses from existing responses and new responses.

    Args:
        responses_by_qid: Existing responses by question ID
        new_responses: New responses to add (item, response)
        model_id: Model ID
        instr_id: Instruction ID
        ds_params: Dataset parameters
        sampling_params: Sampling parameters

    Returns:
        CotResponses object
    """
    # Start with existing responses if any
    responses: dict[str, dict[str, MathResponse | AtCoderResponse | str]] = {}
    if responses_by_qid is not None:
        responses = {qid: dict(resp) for qid, resp in responses_by_qid.items()}

    # Add new responses
    fsp_by_resp_id: dict[str, str] | None = None
    for q_resp_id, response, fsp in new_responses:
        if not response:
            continue
        if q_resp_id.qid not in responses:
            responses[q_resp_id.qid] = {}
        responses[q_resp_id.qid][q_resp_id.uuid] = response

        if fsp is not None:
            if fsp_by_resp_id is None:
                fsp_by_resp_id = {}
            fsp_by_resp_id[q_resp_id.uuid] = fsp

    return CotResponses(
        responses_by_qid=responses,
        fsp_by_resp_id=fsp_by_resp_id,
        model_id=model_id,
        instr_id=instr_id,
        ds_params=ds_params,
        sampling_params=sampling_params,
    )
