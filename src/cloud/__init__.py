from src.cloud.models import CloudModels, NeuralNetCloudModels, TabularCloudModels, EnsembleCloudModels

CLOUD_MODELS = {
    NeuralNetCloudModels.name: NeuralNetCloudModels,
    TabularCloudModels.name: TabularCloudModels,
    EnsembleCloudModels.name: EnsembleCloudModels
}