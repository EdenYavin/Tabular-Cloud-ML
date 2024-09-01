from openai import azure_endpoint
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
from rtfm.configs import TrainConfig, TokenizerConfig, SerializerConfig
from rtfm.inference_utils import InferenceModel
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import prepare_tokenizer
from sklearn.metrics import accuracy_score, f1_score
import torch, os

from langchain_openai.chat_models import AzureChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from src.utils.constansts import MODELS_PATH, SYSTEM_PREDICTION_PROMPT, USER_PREDICTION_PROMPT
from src.cloud.base import CloudModels
from src.utils.helpers import load_prompt

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
            train_config.model_name, device_map="auto", config=config).to(self.device)

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

    def predict(self, X):



        predictions = []
        for x in X:
            output = self.inference_model.predict(
                target_example=x,
                target_colname=self.target_choices,
                target_choices=self.target_choices,
            )
            predictions.append(output)
        return predictions

    def fit(self, X_train, y_train , **kwargs):
        self.target_choices = kwargs.get("labels")
        self.target_column = kwargs.get("target_column")

    def evaluate(self, X, y) -> tuple:
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')
        return accuracy, f1

class ChatGPTCloudModel(CloudModels):

    name = "gpt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.models = AzureChatOpenAI(
            deployment_name="gpt-4o-deployment",
            temperature=0,
            openai_api_version="2024-02-01",
            api_key = os.getenv("OPENAI_API_KEY"),
            azure_endpoint="https://function-app-open-ai-prod-apim.azure-api.net/proxy-api"
        )

        class Prediction(BaseModel):
            llm_prediction: int = Field(description="The label prediction for the given sample data")

        self.models = self.models.with_structured_output(Prediction)
        self.system_prompt = load_prompt(SYSTEM_PREDICTION_PROMPT)
        self.user_prompt = load_prompt(USER_PREDICTION_PROMPT)

    def fit(self, X_train, y_train, **kwargs):
        self.labels = kwargs.get("labels")


    def evaluate(self, X, y) -> tuple:
        return -1, -1

    def predict(self, X):

        formatted_labels = [f"Label {i}: {label}" for i, label in enumerate(self.labels)]

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt)
        ])
        chain = prompt | self.cloud_models

        prediction = chain.invoke(dict(sample=X, labels=formatted_labels))

        return prediction.llm_prediction









