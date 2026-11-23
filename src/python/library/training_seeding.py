"""Seed allocation shared by training and multi-training (no simulator imports)."""
MAX_SEED = 2**32 - 1


def validate_seed(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SEED:
        raise ValueError('Training seed must be an integer between 0 and 2**32 - 1, or null.')
    return value


def seed_block_size(config):
    workers = config.get('env', {}).get('instances', 1)
    evaluations = config.get('train', {}).get('eval_instances', 1)
    for value in (workers, evaluations):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Seed allocation requires positive training and evaluation worker counts.')
    # Evaluation ranks begin at training worker count + 1 (one reserved gap).
    return workers + 1 + evaluations


def training_seed(config):
    seed = validate_seed(config.get('train', {}).get('seed'))
    if seed is not None:
        validate_seed(seed + seed_block_size(config) - 1)
    return seed


def allocate_stage_seeds(configs, starting_seed):
    """Assign disjoint worker seed blocks in execution order; return next free seed."""
    next_seed = validate_seed(starting_seed)
    for config in configs:
        config.setdefault('train', {})['seed'] = next_seed
        training_seed(config)
        if next_seed is not None:
            next_seed += seed_block_size(config)
    return next_seed
