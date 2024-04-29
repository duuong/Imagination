class Generator():
    def __init__(self):
        self.version = 0.1
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
class TrialAlreadyExistsError(Exception):
    def __init__(self, trial_name):
        self.trial_name = trial_name
        super().__init__(f"Trial '{trial_name}' already exists.")