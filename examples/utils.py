def splitStickyStretchy(params, num_samples_per_class=None):
    return [p for p in params if p.obj_name == 'sticky'][:num_samples_per_class], \
        [p for p in params if p.obj_name == 'stretchy'][:num_samples_per_class]
