import tornado


class ExperimentHandler:

    def __init__(self, experiment_config: dict):
        self.experiment_name = experiment_config.get("name")
        self.n_pred_vectors = experiment_config.get("n_pred_vectors")
        self.n_noise_samples = experiment_config.get("n_noise_samples")

    def run_experiment(self):
        pass
