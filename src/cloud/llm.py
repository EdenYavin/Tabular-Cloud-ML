from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.nn import functional as F

from src.utils.cache import Cache
import src.utils.constansts as consts
from src.cloud.base import CloudModel


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SequenceClassificationLLMCloudModel(CloudModel):
    name = "sequence_classification_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to("cpu")
        self.labels = None
        self.output_logits = kwargs.get(consts.LLM_PRED_VECTOR_TYPE_CONFIG_TOKEN, True)
        self.top_k = kwargs.get(consts.LLM_TOP_K_CONFIG_TOKEN, 30000)
        self.cache = Cache(cache_file="sequence_classification_llm.cache")

    def get_probabilities(self, context):
        logits = self.get_logits(context)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def get_logits(self, context):
        inputs = self.tokenizer.encode(context, return_tensors='pt').to("cpu")
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.logits

    def predict(self, X):
        predictions = []
        for x in X:
            x = [round(elem, 10) for elem in x]
            context = ",".join([str(i) for i in x])
            if self.output_logits:
                outputs = self.get_logits(context)
            else:
                outputs = self.get_probabilities(context)
            outputs = outputs.cpu()
            top_k = min(self.top_k, outputs.shape[1])
            top_k_values, top_k_indices = torch.topk(outputs, top_k, largest=True, sorted=True)
            top_k_values = top_k_values.numpy()[0]
            predictions.append(top_k_values)
        return predictions

    def fit(self, X_train, y_train, **kwargs):
        self.labels = kwargs.get("labels")

    def evaluate(self, X, y) -> tuple:
        return -1, -1


class CasualLLMCloudModel(CloudModel):

    name = "casual_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "HuggingFaceTB/SmolLM-360M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.labels = None
        self.output_logits = kwargs.get(consts.LLM_PRED_VECTOR_TYPE_CONFIG_TOKEN, True)
        self.top_k = kwargs.get(consts.LLM_TOP_K_CONFIG_TOKEN, 30000)
        self.cache = Cache(cache_file="casual_llm.cache")

    def get_probabilities(self, context):
        logits = self.get_logits(context)

        # Softmax the logits into probabilities
        probabilities = F.softmax(logits, dim=-1)

        return probabilities

    def get_logits(self, context):

        inputs = self.tokenizer.encode(context, return_tensors='pt').to(device)

        # Get logits from last layer
        with torch.no_grad():
            last_layer_logits = self.model(inputs).logits[:, -1, :]
        return last_layer_logits


    def predict(self, X):

        predictions = []
        for x in X:
            x = [round(elem,5) for elem in x]

            context = ",".join([str(i) for i in x])


            if self.output_logits:
                outputs = self.get_logits(context)
            else:
                outputs = self.get_probabilities(context)

            # Get the TOP_K values and their indices
            outputs = outputs.cpu()
            top_k = min(self.top_k, outputs.shape[1])
            top_k_values, top_k_indices = torch.topk(outputs, top_k, largest=True, sorted=True)

            top_k_values = top_k_values.numpy()[0]

            predictions.append(top_k_values)

        return predictions

    def fit(self, X_train, y_train , **kwargs):
        self.labels = kwargs.get("labels")


    def evaluate(self, X, y) -> tuple:
        return -1, -1


class MaskedLLMCloudModel(CloudModel):

    name = "masked_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "google-bert/bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.labels = None
        self.output_logits = kwargs.get(consts.LLM_PRED_VECTOR_TYPE_CONFIG_TOKEN, True)
        self.top_k = kwargs.get(consts.LLM_TOP_K_CONFIG_TOKEN, 30000)
        self.cache = Cache(cache_file="casual_llm.cache")

    def get_probabilities(self, context):
        logits = self.get_logits(context)

        # Softmax the logits into probabilities
        probabilities = F.softmax(logits, dim=-1)

        return probabilities

    def get_logits(self, context):

        context += " [MASK]"

        inputs = self.tokenizer(context, return_tensors='pt').to(device)

        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1][0]

        # Get predictions
        with torch.no_grad():
            # inputs = inputs.to(torch.int32)
            outputs = self.model(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :].squeeze()

        return mask_token_logits

    def predict(self, X):

        predictions = []
        for x in X:
            x = [round(elem,2) for elem in x]
            x.append("[MASK]")
            context = str(x)

            if self.output_logits:
                outputs = self.get_logits(context)
            else:
                outputs = self.get_probabilities(context)

            # Get the TOP_K values and their indices
            outputs = outputs.cpu()
            top_k = min(self.top_k, outputs.shape[0])
            top_k_values, top_k_indices = torch.topk(outputs, top_k, largest=True, sorted=True)

            top_k_values = top_k_values.numpy()


            predictions.append(top_k_values)

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









