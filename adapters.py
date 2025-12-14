from regex import F
import torch
from typing import Literal, Callable, List, Tuple, Dict, Optional

from transformers import PreTrainedTokenizerBase

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by group size.

    Returns:
        advantages: shape (rollout_batch_size,)
        raw_rewards: shape (rollout_batch_size,)
        metadata: dict of scalar stats
    """

    # 一些基本检查（方便 debug）
    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size == len(repeated_ground_truths), \
        "rollout_responses 和 repeated_ground_truths 长度必须一致"
    assert rollout_batch_size % group_size == 0, \
        "batch_size 必须能被 group_size 整除"

    num_groups = rollout_batch_size // group_size

    # 1. 计算每个 rollout 的 reward（以及你想 log 的其他分量）
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []

    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(resp, gt)
        raw_rewards_list.append(float(scores["reward"]))
        # 下面这两个字段可选：如果 test 里没用，可以不记
        if "format_reward" in scores:
            format_rewards_list.append(float(scores["format_reward"]))
        if "answer_reward" in scores:
            answer_rewards_list.append(float(scores["answer_reward"]))

    # 转成 tensor，方便做分组运算
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)  # (B,)

    # 2. 按组 reshape 成 (num_groups, group_size)
    rewards_group = raw_rewards.view(num_groups, group_size)  # (G, group_size)

    # 每组的均值
    group_mean = rewards_group.mean(dim=1, keepdim=True)  # (G, 1)

    if normalize_by_std:
        # 每组的标准差（unbiased=False 对应“总体”方差）
        group_std = rewards_group.std(dim=1, keepdim=True, unbiased=True)
        # 避免除零：加上 advantage_eps
        advantages_group = (rewards_group - group_mean) / (group_std + advantage_eps)
    else:
        # 只减均值，不除标准差
        advantages_group = rewards_group - group_mean

    # 展平成 (batch_size,)
    advantages = advantages_group.view(-1)

    # 3. 构造一些元数据，用于 log
    metadata: Dict[str, float] = {
        "raw_reward_mean": raw_rewards.mean().item(),
        "raw_reward_std": raw_rewards.std(unbiased=False).item(),
        "raw_reward_min": raw_rewards.min().item(),
        "raw_reward_max": raw_rewards.max().item(),
    }

    if format_rewards_list:
        fr = torch.tensor(format_rewards_list, dtype=torch.float32)
        metadata["format_reward_mean"] = fr.mean().item()
        metadata["format_reward_std"] = fr.std(unbiased=True).item()

    if answer_rewards_list:
        ar = torch.tensor(answer_rewards_list, dtype=torch.float32)
        metadata["answer_reward_mean"] = ar.mean().item()
        metadata["answer_reward_std"] = ar.std(unbiased=True).item()

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,      # (B, 1)
    policy_log_probs: torch.Tensor,              # (B, T)
) -> torch.Tensor:
    """
    Naive per-token policy gradient loss:
        loss_{b,t} = - A_b * logpi_{b,t}
    where A_b is scalar reward/advantage for rollout b.
    """
    # 确保 dtype 一致
    advantages = raw_rewards_or_advantages.to(policy_log_probs.dtype)  # (B, 1)

    # 依靠广播扩展到 (B, T)，并计算逐 token 的 PG loss
    per_token_loss = -advantages * policy_log_probs                   # (B, T)

    return per_token_loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,        # (B, 1)
    policy_log_probs: torch.Tensor,  # (B, T) new policy
    old_log_probs: torch.Tensor,     # (B, T) old policy
    cliprange: float,                # epsilon
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 确保 dtype 一致
    advantages = advantages.to(policy_log_probs.dtype)  # (B, 1)
    # 广播到每个 token
    advantages = advantages.expand_as(policy_log_probs) # (B, T)

    # 1) 计算概率比 r_t = pi_new / pi_old
    log_ratio = policy_log_probs - old_log_probs        # (B, T)
    ratio = torch.exp(log_ratio)                        # (B, T)

    # 2) 未剪切的 PG 目标：r_t * A_t
    pg_unclipped = ratio * advantages                   # (B, T)

    # 3) 剪切后的 PG 目标：clip(r_t, 1-e, 1+e) * A_t
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_clipped = clipped_ratio * advantages             # (B, T)

    # 4) 取 min，然后加负号变成 loss
    pg_objective = torch.minimum(pg_unclipped, pg_clipped)  # (B, T)
    loss = -pg_objective                                   # (B, T)

    # 5) metadata：标记哪些 token 实际用了剪切后的分支
    is_clipped = pg_clipped < pg_unclipped                 # (B, T), bool

    metadata = {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "is_clipped": is_clipped,
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    选择并计算指定的 policy-gradient loss。
    返回:
        loss:   (batch_size, sequence_length) 的 per-token loss
        metadata: dict, 里面放一些统计信息（比如 clip fraction）
    """

    # 基本 sanity check
    assert loss_type in {"no_baseline", "reinforce_with_baseline", "grpo_clip"}, \
        f"Unknown loss_type: {loss_type}"

    # 统一的 metadata 容器
    metadata: Dict[str, torch.Tensor] = {}

    if loss_type == "no_baseline":
        # A = raw_rewards
        assert raw_rewards is not None, \
            "raw_rewards is required when loss_type == 'no_baseline'"
        # raw_rewards: (B, 1), policy_log_probs: (B, T)
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        # naive 版本一般没有额外统计，可以留空 dict

    elif loss_type == "reinforce_with_baseline":
        # A = advantages（已经做过 group-normalization 的 reward）
        assert advantages is not None, \
            "advantages is required when loss_type == 'reinforce_with_baseline'"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        # 同样 metadata 可以先留空，或者你也可以加个 mean/std
        # metadata["adv_mean"] = advantages.mean()
        # metadata["adv_std"] = advantages.std(unbiased=False)

    elif loss_type == "grpo_clip":
        # GRPO-Clip: 需要 A、old_log_probs、cliprange
        assert advantages is not None, \
            "advantages is required when loss_type == 'grpo_clip'"
        assert old_log_probs is not None, \
            "old_log_probs is required when loss_type == 'grpo_clip'"
        assert cliprange is not None, \
            "cliprange is required when loss_type == 'grpo_clip'"

        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages=advantages,              # (B, 1)
            policy_log_probs=policy_log_probs,  # (B, T)
            old_log_probs=old_log_probs,        # (B, T)
            cliprange=cliprange,                # float
        )
        # 把内部的统计原封不动地往外转发
        metadata.update(grpo_metadata)

    return loss, metadata
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the mean of `tensor` over the given dimension, considering only
    positions where `mask == 1`.

    Args:
        tensor: data tensor.
        mask:  same shape as `tensor`; 1 / True means "include", 0 / False means "ignore".
        dim:   dimension to reduce. If None, reduce over all elements.

    Returns:
        Masked mean. Shape follows `tensor.mean(dim)` semantics:
          - dim is None -> scalar
          - dim is int  -> tensor with that dimension removed
    """
    # 1. 形状检查
    assert tensor.shape == mask.shape, \
        f"tensor.shape {tensor.shape} and mask.shape {mask.shape} must match"

    # 2. 类型对齐
    mask = mask.to(tensor.dtype)

    if dim is None:
        # 对所有 mask==1 的元素整体求平均
        masked_sum = (tensor * mask).sum()   # 标量
        count = mask.sum()                   # 标量，可能为 0
        return masked_sum / count           # 0/0 -> nan（这是期望行为）
    else:
        # 在指定维度上做 masked mean
        masked_sum = (tensor * mask).sum(dim=dim)   # 形状：去掉 dim
        count = mask.sum(dim=dim)                   # 同形状，可能为 0
        return masked_sum / count                  # 对应位置全 0 时 -> nan

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行一次 GRPO 的 microbatch 训练步骤：
      - 计算 per-token policy gradient loss
      - 用 response_mask 做 masked_mean 得到每个样本的 loss
      - 对 batch 做平均，除以 gradient_accumulation_steps
      - backward() 把梯度累积到参数上
    """

    assert gradient_accumulation_steps >= 1, \
        f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}"

    # 1. 计算每个 token 的 PG loss（不管是 no_baseline / baseline / grpo_clip）
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,   # (B, T)
        loss_type=loss_type,
        raw_rewards=raw_rewards,             # (B, 1) or None
        advantages=advantages,               # (B, 1) or None
        old_log_probs=old_log_probs,         # (B, T) or None
        cliprange=cliprange,                 # float or None
    )
    # per_token_loss: (B, T)

    # 2. 只在 response token 上做平均，得到每个样本的 loss: (B,)
    per_example_loss = masked_mean(
        tensor=per_token_loss,
        mask=response_mask,   # (B, T)，1 表示 response token
        dim=1,                # 沿 sequence 维度做 mean
    )  # shape: (B,)

    # 3. 再对 batch 做平均，得到 microbatch 的 scalar loss
    microbatch_loss = per_example_loss.mean()  # scalar

    # 4. 为了梯度累积，按 steps 做缩放
    microbatch_loss = microbatch_loss / gradient_accumulation_steps
 
    # 5. 反向传播，把这一小批的梯度加到参数的 .grad 上
    microbatch_loss.backward()

    # 6. 准备 metadata：保留底层 loss 的统计，并加上本步的 loss 方便 log
    metadata: Dict[str, torch.Tensor] = dict(loss_metadata)
    # 一般 logging 只需要一个 detached 的标量
    metadata["microbatch_loss"] = microbatch_loss.detach()

    return microbatch_loss, metadata

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_input_ids = []
    batch_response_masks = []
    prompt_and_output_lens = []

    for prompt, output in zip(prompt_strs, output_strs):
        # 分别分词
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        # 拼接
        input_ids = prompt_ids + output_ids
        prompt_and_output_lens.append(len(input_ids))

        # response_mask: prompt部分0, output部分1
        response_mask = [0] * len(prompt_ids) + [1] * len(output_ids)

        batch_input_ids.append(input_ids)
        batch_response_masks.append(response_mask)

    # 对齐 padding
    max_len = max(prompt_and_output_lens)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def pad_to_max(seq, pad_value):
        return seq + [pad_value] * (max_len - len(seq))

    batch_input_ids = [pad_to_max(x, pad_token_id) for x in batch_input_ids]
    batch_response_masks = [pad_to_max(x, 0) for x in batch_response_masks]

    # 转为 tensor
    input_ids = torch.tensor(batch_input_ids)
    response_mask = torch.tensor(batch_response_masks)

    # 生成 labels（右移一个 token）
    labels = input_ids.clone()

    # 根据题意：去掉最后一个 token 作为输入，去掉第一个 token 作为输出
    input_ids = input_ids[:, :-1]
    labels = labels[:, 1:]
    response_mask = response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    """
    Compute per-token entropy over vocabulary dimension.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
    
    Returns:
        Tensor of shape (batch_size, seq_len) — entropy of each next-token prediction
    """
    # 1. logsumexp for numerical stability: log(∑ exp(logits))
    logsumexp = torch.logsumexp(logits, dim=-1)  # shape: (B, S)
    
    # 2. softmax probabilities
    probs = torch.softmax(logits, dim=-1)        # shape: (B, S, V)
    
    # 3. expected value of logits under probs
    expected_logits = torch.sum(probs * logits, dim=-1)  # shape: (B, S)
    
    # 4. entropy = logsumexp - expected_logits
    entropy = logsumexp - expected_logits
    
    return entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # 1. 前向传播获取 logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # 2. 计算每个位置的 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)

    # 3. 取出 label 对应的 log-prob
    response_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # 4. 若需要计算熵
    result = {"log_probs": response_log_probs}
    if return_token_entropy:
        token_entropy = run_compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result

