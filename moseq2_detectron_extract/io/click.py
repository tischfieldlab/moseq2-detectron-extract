from typing import Dict, Optional, Type

import click
import ruamel.yaml as yaml

from moseq2_detectron_extract.io.util import read_yaml


class OptionalParamType(click.ParamType):
    ''' Wrap a `click.ParamType` and make it optional'''
    def __init__(self, param_type: click.ParamType):
        self.param_type = param_type
        self.name = f'{param_type.name} (Optional)'

    def convert(self, value, param, ctx):
        if not value:
            return
        return self.param_type.convert(value, param, ctx)


def click_param_annot(click_cmd: click.Command) -> Dict[str, Optional[str]]:
    ''' Given a click.Command instance, return a dict that maps option names to help strings.
    Currently skips click.Arguments, as they do not have help strings.

    Parameters:
    click_cmd (click.Command): command to introspect

    Returns:
    annotations (dict): click.Option.human_readable_name as keys; click.Option.help as values
    '''
    annotations = {}
    for param in click_cmd.params:
        if isinstance(param, click.Option):
            annotations[param.human_readable_name] = param.help
    return annotations


def click_monkey_patch_option_show_defaults():
    ''' Monkey patch click.core.Option to turn on showing default values.
    '''
    orig_init = click.core.Option.__init__
    def new_init(self, *args, **kwargs):
        ''' This version of click.core.Option.__init__ will set show default values to True
        '''
        orig_init(self, *args, **kwargs)
        self.show_default = True
    # end new_init()
    click.core.Option.__init__ = new_init # type: ignore


def get_command_defaults(command: click.Command, skip_required: bool = False):
    """Get the defualt values for the options of `command`.

    Args:
        command: click command
        skip_required: if True, skip options which are marked as required

    Returns:
        Default arguments for command.
    """
    out = {}
    for item in command.params:
        if item.name is not None and item.name.startswith('fake_'):
            # option groups will create entries with `fake_<long-random-string>`
            # we should skip these
            continue

        if item.required and skip_required:
            continue

        out[item.name] = item.default

    return out


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name: str) -> Type[click.Command]:
    """Get a command class which supports configuration files.

        Create and return a class inheriting `click.Command` which accepts a configuration file
        containing arguments/options accepted by the command.

        The returned class should be passed to the `@click.Commnad` parameter `cls`:

        ```
        @cli.command(name='command-name', cls=command_with_config('config_file'))
        ```

    Args:
        config_file_param_name (str): name of the parameter that accepts a configuration file

    Returns:
        class (Type[click.Command]): Class to use when constructing a new click.Command
    """

    class CustomCommandClass(click.Command):
        """Command which accepts config file.

        Methods:
            invoke: invoke the command
        """

        def invoke(self, ctx):
            """Invoke the command accepting config as part of argument.

            Args:
                ctx: click.Command arguments including config file.
            """
            # grab the config file
            config_file = ctx.params[config_file_param_name]
            param_defaults = {p.human_readable_name: p.default for p in self.params if isinstance(p, click.core.Option) and not p.name.startswith('fake_')}
            param_defaults = {k: tuple(v) if isinstance(v, list) else v for k, v in param_defaults.items()}
            param_cli = {k: tuple(v) if isinstance(v, list) else v for k, v in ctx.params.items()}

            if config_file is not None:
                config_data = read_yaml(config_file)
                # modified to only use keys that are actually defined in options
                config_data = {
                    k: tuple(v) if isinstance(v, yaml.comments.CommentedSeq) else v
                    for k, v in config_data.items()
                    if k in param_defaults.keys()
                }

                # find differences btw config and param defaults
                diffs = set(param_defaults.items()) ^ set(param_cli.items())

                # combine defaults w/ config data
                combined = {**param_defaults, **config_data}

                # update cli params that are non-default
                keys = [d[0] for d in diffs]
                for k in set(keys):
                    combined[k] = ctx.params[k]

                ctx.params = combined

            return super().invoke(ctx)

    return CustomCommandClass
