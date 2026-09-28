import torch
from mmengine.hooks import Hook

from mmpose.registry import HOOKS


@HOOKS.register_module()
class OptHookA(Hook):

    priority = 'NORMAL'

    def before_train(self, runner) -> None:
        runner.model.to(memory_format=torch.channels_last)
        runner.logger.info(
            'OptHookA: on')
