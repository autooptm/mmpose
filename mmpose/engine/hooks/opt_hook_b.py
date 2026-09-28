from typing import List, Optional

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmpose.registry import HOOKS


@HOOKS.register_module()
class OptHookB(Hook):

    priority = 'LOW'

    def __init__(self,
                 targets: Optional[List[str]] = None,
                 mode: Optional[str] = 'reduce-overhead',
                 dynamic: bool = False,
                 stock_eval: bool = True):
        self.targets = list(targets) if targets else ['backbone']
        self.mode = mode
        self.dynamic = dynamic
        self.stock_eval = stock_eval
        self._stock = {}
        self._wrapped = {}

    @staticmethod
    def _bare(model):
        return model.module if is_model_wrapper(model) else model

    def before_train(self, runner) -> None:
        model = self._bare(runner.model)
        for name in self.targets:
            sub = getattr(model, name, None)
            if sub is None:
                raise AttributeError(
                    f'OptHookB: model has no submodule {name!r}')
            self._stock[name] = sub
            setattr(
                model, name,
                torch.compile(sub, mode=self.mode, dynamic=self.dynamic))
            wrapped = getattr(model, name)
            assert getattr(wrapped, '_orig_mod', None) is not None, (
                f'OptHookB: could not wrap {name!r}')
            runner.logger.info(
                f'OptHookB: wrapped {name} '
                f'(mode={self.mode}, dynamic={self.dynamic}, '
                f'stock_eval={self.stock_eval})')

    def _use_stock(self, runner, on: bool) -> None:
        if not self.stock_eval or not self._stock:
            return
        model = self._bare(runner.model)
        for name, eager in self._stock.items():
            current = getattr(model, name)
            if on:
                self._wrapped[name] = current
                setattr(model, name, eager)
            else:
                setattr(model, name, self._wrapped[name])

    def before_val_epoch(self, runner) -> None:
        self._use_stock(runner, True)

    def after_val_epoch(self, runner, metrics=None) -> None:
        self._use_stock(runner, False)

    def before_test_epoch(self, runner) -> None:
        self._use_stock(runner, True)

    def after_test_epoch(self, runner, metrics=None) -> None:
        self._use_stock(runner, False)
