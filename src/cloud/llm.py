import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json, re, torch

from torch.xpu import device
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

from src.utils.cache import Cache
from src.utils.constansts import MODELS_PATH, SYSTEM_PREDICTION_PROMPT, USER_PREDICTION_PROMPT
from src.cloud.base import CloudModels
from src.utils.helpers import load_prompt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class BertCloudModel(CloudModels):

    name = "bert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
        self.models = AutoModelForMaskedLM.from_pretrained("google-bert/bert-large-uncased").to(device)

        self.labels = None

    def get_word_probabilities(self, context, mask_token='[MASK]', top_k=None):
        # Ensure the context ends with the mask token
        if not context.endswith(mask_token):
            context += f" {mask_token}"

        # Tokenize the input with truncation
        inputs = self.tokenizer(context, return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length).to(device)
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        # Get predictions
        with torch.no_grad():
            outputs = self.models(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :].squeeze()

        # Convert the logits tensor to a NumPy array
        mask_token_logits_np = mask_token_logits.cpu().numpy()

        return mask_token_logits_np

    def predict(self, X):

        predictions = []
        for x in X:
            x = [round(elem,10) for elem in x]
            output = self.get_word_probabilities(context=str(x), top_k=1000)
            predictions.append(output)
        return predictions

    def fit(self, X_train, y_train , **kwargs):
        self.labels = kwargs.get("labels")


    def evaluate(self, X, y) -> tuple:
        return -1, -1

# class OllamaCloudModel(CloudModels):
#
#     name = "ollama"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         self.models = ChatOllama(model="llama3.1",temperature=0.1)
#         self.system_prompt = load_prompt(SYSTEM_PREDICTION_PROMPT)
#         self.user_prompt = load_prompt(USER_PREDICTION_PROMPT)
#         self.labels, self.dataset_description = None, None
#         self.cache = Cache(cache_file=f"{OllamaCloudModel.name}.pkl")
#
#     def fit(self, X_train, y_train, **kwargs):
#         self.labels = kwargs.get("labels")
#         self.dataset_description = kwargs.get("dataset_description")
#
#     def evaluate(self, X, y) -> tuple:
#         X = X.head(100)
#         y = y.head(100)
#         predictions = []
#         for x in tqdm(X.iterrows(), total=len(X), leave=True, position=0):
#             y_hat = self.predict(x)
#             predictions.append(np.argmax(y_hat))
#
#         accuracy = accuracy_score(y, predictions)
#         f1 = f1_score(y, predictions, average='weighted')
#         return accuracy, f1
#
#     def _get_prediction(self, chain, X, labels, dataset_description):
#         answer = chain.invoke(dict(sample=X, labels=labels, dataset_description=dataset_description))
#
#         try:
#             answer = json.loads(answer)
#             prediction = answer["label"]
#             prediction = int(prediction)
#
#         except Exception:
#             prediction = re.findall(r"\d", answer)[0]
#             prediction = int(prediction)
#
#         return prediction
#
#     def predict(self, X):
#
#         answer = self.cache.get(str(X))
#         if answer is not None:
#             return answer
#
#         formatted_labels = [f"{i}: {label}" for i, label in enumerate(self.labels)]
#
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", self.system_prompt),
#             ("user", self.user_prompt)
#         ])
#         chain = prompt | self.models | StrOutputParser()
#
#         prediction = self._get_prediction(chain, X, formatted_labels, self.dataset_description)
#
#         while prediction >= len(self.labels):
#             print(f"{prediction} is bigger then Labels size: {len(self.labels)}, TRYING AGAIN")
#             prediction = self._get_prediction(chain, X, formatted_labels, self.dataset_description)
#
#         one_hot_prediction = np.zeros(len(self.labels))
#         one_hot_prediction[prediction] = 1
#
#         self.cache.set(str(X), one_hot_prediction)
#         return [one_hot_prediction]



# class ChatGPTCloudModel(CloudModels):
#
#     name = "gpt"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         self.models = AzureChatOpenAI(
#             deployment_name="gpt-4o-deployment",
#             temperature=0,
#             openai_api_version="2024-02-01",
#         )
#
#         class Prediction(BaseModel):
#             llm_prediction: int = Field(description="The label prediction for the given sample data")
#
#         self.models = self.models.with_structured_output(Prediction)
#         self.system_prompt = load_prompt(SYSTEM_PREDICTION_PROMPT)
#         self.user_prompt = load_prompt(USER_PREDICTION_PROMPT)
#
#     def fit(self, X_train, y_train, **kwargs):
#         self.labels = kwargs.get("labels")
#
#
#     def evaluate(self, X, y) -> tuple:
#         return -1, -1
#
#     def predict(self, X):
#
#         formatted_labels = [f"Label {i}: {label}" for i, label in enumerate(self.labels)]
#
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", self.system_prompt),
#             ("user", self.user_prompt)
#         ])
#         chain = prompt | self.models
#
#         answer = chain.invoke(dict(sample=X, labels=formatted_labels))
#         prediction = answer.llm_prediction
#         one_hot_prediction = np.zeros(len(self.labels))
#         one_hot_prediction[prediction] = 1
#
#         return [one_hot_prediction]









