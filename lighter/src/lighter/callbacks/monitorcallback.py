from lighter.callbacks import Callback

class MonitorCallback(Callback):
    '''
    Base class for callbacks that monitor a quantity and evaluates improvements.
    '''
    def __init__(
        self,
        monitor="val_loss",
        mode="auto",
        baseline=None,
        min_delta=0,
    ):
        super().__init__()
        if mode not in ["auto", "min", "max"]:
            raise ValueError('"mode" must be ["auto", "min", "max"]')

        self.monitor    = monitor
        self.mode       = mode
        self.best       = baseline
        self.min_delta  = abs(min_delta)
        self._set_monitor_op()

    def _set_monitor_op(self):
        less = lambda x, y: x < y
        greater = lambda x, y: x > y

        if self.mode == "min":
            self.monitor_op = less
        elif self.mode == "max":
            self.monitor_op = greater
        else:
            if 'loss' in self.monitor:
                self.monitor_op = less

    def _is_improvement(self, monitor_value, reference_value):
        if reference_value is None:
            return True
        return self.monitor_op(monitor_value - self.min_delta, reference_value)