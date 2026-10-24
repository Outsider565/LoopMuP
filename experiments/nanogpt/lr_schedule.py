"""Step-based learning-rate schedules: warmup-stable-decay ('wsd') and warmup + cosine ('cosine')."""
import math


def scheduled_lr(step, total_steps, warmup_steps, decay_steps, peak_lr, min_lr=0.0, schedule='wsd'):
    """
    LR of the 0-based optimizer step `step`. Both schedules warm up linearly over warmup_steps.
    'wsd' then holds peak_lr and decays linearly over the final decay_steps (0: no decay), from peak_lr at the
    first decay step to min_lr + (peak_lr - min_lr) / decay_steps at the last step (min_lr at step total_steps);
    'cosine' decays from peak_lr to min_lr along a half cosine and ignores decay_steps.
    """
    if schedule not in ('wsd', 'cosine'):
        raise ValueError(f'unknown LR schedule: {schedule}')
    used_decay = decay_steps if schedule == 'wsd' else 0
    if total_steps <= 0 or warmup_steps < 0 or decay_steps < 0 or warmup_steps + used_decay > total_steps:
        raise ValueError('invalid total_steps, warmup_steps or decay_steps')
    if step < 0:
        raise ValueError('step must be nonnegative')
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if schedule == 'wsd':
        decay_start = total_steps - decay_steps
        if decay_steps == 0 or step < decay_start:
            return peak_lr
        progress = min(1.0, (step - decay_start) / decay_steps)
        return min_lr + (peak_lr - min_lr) * (1.0 - progress)
    if total_steps == warmup_steps:
        return peak_lr
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
