from src.cloud.models import CloudModels, NeuralNetCloudModels, TabularCloudModels, EnsembleCloudModels, ImageCloudModels, ResNetEmbeddingCloudModel

CLOUD_MODELS = {
    NeuralNetCloudModels.name: NeuralNetCloudModels,
    TabularCloudModels.name: TabularCloudModels,
    EnsembleCloudModels.name: EnsembleCloudModels,
    ImageCloudModels.name: ImageCloudModels,
    ResNetEmbeddingCloudModel.name: ResNetEmbeddingCloudModel
}