
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
from rtfm.configs import TrainConfig, TokenizerConfig, SerializerConfig
from rtfm.inference_utils import InferenceModel
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import prepare_tokenizer
from sklearn.metrics import accuracy_score, f1_score
import torch

from src.utils.constansts import MODELS_PATH
from src.cloud.base import CloudModels


class TabularLLMCloudModel(CloudModels):

    name = "tabular_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Configuration setup
        train_config = TrainConfig(model_name="mlfoundations/tabula-8b", context_length=8192)
        tokenizer_config = TokenizerConfig()
        serializer_config = SerializerConfig()

        # Load the configuration
        config = AutoConfig.from_pretrained(train_config.model_name)

        # Set the torch_dtype to bfloat16 which matches TabuLa train/eval setup
        config.torch_dtype = torch.bfloat16

        # Device setup
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load model and tokenizer
        self.model = LlamaForCausalLM.from_pretrained(
            train_config.model_name, device_map="auto", config=config, cache_dir=MODELS_PATH).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, cache_dir=MODELS_PATH)
        self.serializer = get_serializer(serializer_config)

        # Prepare tokenizer
        self.tokenizer, self.model = prepare_tokenizer(
            self.model,
            tokenizer=self.tokenizer,
            pretrained_model_name_or_path=train_config.model_name,
            model_max_length=train_config.context_length,
            use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
            serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
            serializer_tokens=self.serializer.special_tokens
            if tokenizer_config.add_serializer_tokens
            else None,
        )

        # Initialize inference model
        self.inference_model = InferenceModel(model=self.model, tokenizer=self.tokenizer, serializer=self.serializer)

        print(f"LLM Using device: {self.device}")

    def predict(self, X, **kwargs):

        target_choices = kwargs.get("labels")
        target_column = kwargs.get("target_column")

        predictions = []
        for x in X:
            output = self.inference_model.predict(
                target_example=x,
                target_colname=target_column,
                target_choices=target_choices,
            )
            predictions.append(output)
        return predictions

    def fit(self, X_train, y_train):
        pass

    def evaluate(self, X, y) -> tuple:
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')
        return accuracy, f1