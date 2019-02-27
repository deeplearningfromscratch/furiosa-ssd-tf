import tensorflow as tf
slim = tf.contrib.slim


def get_init_fn_mobilenet(flags, name_remap=None):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if tf.train.latest_checkpoint(flags.train_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s.' % flags.train_dir)
        return None
    exclusion_scopes = []
    if flags.checkpoint_exclude_scopes:
        exclusion_scopes = [scope.strip() for scope in flags.checkpoint_exclude_scopes.split(',')]
        tf.logging.info('Exclusion scopes for restoring from checkpoint %s.' % (exclusion_scopes))

    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):

        excluded = False
        for exclusion in exclusion_scopes:
            if exclusion in var.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if flags.checkpoint_model_scope is not None:
        if flags.checkpoint_model_scope.strip() == '':
            variables_to_restore = {var.op.name.replace(flags.model_name + '/', ''): var for var in variables_to_restore}
        else:
            variables_to_restore = {var.op.name.replace(flags.model_name, flags.checkpoint_model_scope.strip()): var for var in
                                    variables_to_restore}

        if name_remap is not None:
            renamed_variables_to_restore = dict()
            for var_name, var in variables_to_restore.items():
                found = False
                for k, v in name_remap.items():
                    if k in var_name:
                        var_name = var_name.replace(k, v)
                        found = True
                renamed_variables_to_restore[var_name] = var
                if not found:
                    renamed_variables_to_restore[var_name] = var
            variables_to_restore = renamed_variables_to_restore

    checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path) if tf.gfile.IsDirectory(
        flags.checkpoint_path) else flags.checkpoint_path

    tf.logging.info('Fine-tuning from %s.' % (checkpoint_path))
    tf.logging.info('Ignoring missing vars: %s.' % (flags.ignore_missing_vars))

    if not variables_to_restore:
        raise ValueError('variables_to_restore cannot be empty')

    if flags.ignore_missing_vars:

        reader = tf.train.NewCheckpointReader(checkpoint_path)
        if isinstance(variables_to_restore, dict):
            var_dict = variables_to_restore
        else:
            var_dict = {var.op.name: var for var in variables_to_restore}

        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning('Variable %s missing in checkpoint %s.', var, checkpoint_path)

        variables_to_restore = available_vars

    if variables_to_restore:
        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=flags.ignore_missing_vars)
    else:
        tf.logging.warning('No Variables to restore.')
        return None