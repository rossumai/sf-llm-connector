import concurrent.futures
import json
from typing import Any, Dict, List

import boto3
from pydantic import BaseModel, Field, ValidationError, confloat, conint

from rossum_python import *

# Default values that you may want to change
DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # May become deprecated by Amazon
ANTHROPIC_VERSION = "bedrock-2023-05-31"  # May become deprecated by Amazon
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 64


def rossum_hook_request_handler(payload: Dict) -> Dict[str, List[str]]:
    """
    Handles the Rossum hook request, processes the payload, and returns the appropriate output.

    :param payload: Dictionary containing the request payload.
    :return: Replace operations.
    """
    rossum = RossumPython.from_payload(payload)
    payload_configs = payload.get("settings", {}).get("configurations", [])
    payload_secrets = payload.get("secrets", {})

    keep_history = payload.get("settings", {}).get("keep_history", False)
    history = [] if keep_history else None

    for config in payload_configs:
        try:
            config = ConfigModel(**config)
            secrets = SecretsModel(**payload_secrets)

            content_handler = ContentHandler(rossum, config.input, config.batch_size)
            generator = OutputGenerator(config, secrets, content_handler, history if keep_history else None)

            output = generator.generate_output()
            content_handler.update_datapoints(config.output, output)

            if keep_history:
                history = generator.history
        except ValidationError as e:
            rossum.show_error(format_validation_errors(e.errors()))
        except Exception as e:
            rossum.show_error(str(e))

    return rossum.hook_response()


class ConfigModel(BaseModel):
    input: Dict[str, str] = Field(default_factory=Dict)
    prompt: str = Field(min_length=1)
    output: str = Field(min_length=1)
    model: str = Field(default=DEFAULT_MODEL)
    region: str = Field(default=DEFAULT_REGION)
    max_tokens: conint(ge=1) = Field(default=DEFAULT_MAX_TOKENS)
    temperature: confloat(ge=0.0, le=1.0) = Field(default=0.0)
    batch_size: conint(ge=0) = Field(default=1)


class SecretsModel(BaseModel):
    key: str
    secret: str


class ContentHandler:
    """
    Handles the extraction and preparation of data from the Rossum fields for model input.

    :param rossum: The RossumPython instance.
    :param input_pairs: Dictionary mapping descriptors to Rossum field IDs.
    :param batch_size: The size of each batch for processing the input.
    """

    def __init__(self, rossum: RossumPython, input_pairs: Dict[str, str], batch_size: int):
        self.rossum = rossum
        self.batch_size = batch_size
        self._extracted_values = self._extract_values(input_pairs)
        self.user_input = self._generate_batched_input()

    def _extract_values(self, input_pairs: Dict[str, str]) -> Dict[str, Any]:
        """
        Extracts values from Rossum fields based on input pairs.

        :param input_pairs: Dictionary mapping descriptors to Rossum field IDs.
        :return: Dictionary of extracted values.
        """
        extracted_values = {}
        for descriptor, dp_id in input_pairs.items():
            if not hasattr(self.rossum.field, dp_id):
                raise Exception(f"No field found with ID '{dp_id}'")
            field = getattr(self.rossum.field, dp_id)
            try:
                extracted_values[descriptor] = str(field.value)
            except AttributeError:
                extracted_values[descriptor] = list(field.all_values)
        return extracted_values

    def _generate_batched_input(self) -> List[Dict[str, Any]]:
        """
        Generates batched input for the model based on extracted values.

        :return: List of dictionaries, each representing a batch of user input.
        """
        single_values = {k: v for k, v in self._extracted_values.items() if not isinstance(v, list)}
        list_values = {k: v for k, v in self._extracted_values.items() if isinstance(v, list)}

        if len(list_values) == 1:
            key, value = next(iter(list_values.items()))
            single_values[key] = value
        elif len(list_values) > 1:
            single_values["line items"] = self._combine_line_items(list_values)

        return self._split_into_batches(single_values)

    def _split_into_batches(self, user_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits the user input into batches based on the batch size.

        :param user_input: Dictionary containing the user input.
        :return: List of dictionaries, each representing a batch of user input.
        """
        if self.batch_size == 0:
            return [user_input]

        total_items = max((len(v) for v in user_input.values() if isinstance(v, list)), default=1)
        batches = []

        for i in range(0, total_items, self.batch_size):
            batch = {}
            for key, value in user_input.items():
                if isinstance(value, list):
                    batch_slice = value[i: i + self.batch_size]
                    batch[key] = batch_slice if len(batch_slice) > 1 else batch_slice[0]
                else:
                    batch[key] = value
            batches.append(batch)

        return batches

    def update_datapoints(self, output_key: str, output_values: List[str]) -> None:
        """
        Updates the Rossum fields with the generated output.

        :param output_key: The key where the output should be stored.
        :param output_values: List of output values generated by the model.
        """
        try:
            if len(output_values) == 1:
                setattr(self.rossum.field, output_key, output_values[0])
            else:
                for i, row in enumerate(self.rossum.field.line_items):
                    setattr(row, output_key, output_values[i])
        except Exception:
            raise Exception("Amount of datapoints doesn't match amount of responses.")

    @staticmethod
    def _combine_line_items(line_items: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Combines line items into a list of dictionaries.

        :param line_items: Dictionary of line items.
        :return: List of dictionaries representing combined line items.
        """
        keys = list(line_items.keys())
        values = list(line_items.values())
        return [
            {key: value[i] for key, value in zip(keys, values)}
            for i in range(len(values[0]))
        ]


class OutputGenerator:
    """
    Generates the output by interacting with the Bedrock service using the specified model.

    :param config: Configuration model instance.
    :param secrets: Secrets model instance.
    :param content: ContentHandler instance managing input data.
    """

    def __init__(self, config: ConfigModel, secrets: SecretsModel, content: ContentHandler, history: List[Dict[str, str]]):
        self.config = config
        self.secrets = secrets
        self.content = content
        self.history = history
        self.client = self._initialize_client()

    def _initialize_client(self) -> boto3.client:
        """
        Initializes the Bedrock client using the provided secrets and configuration.

        :return: Initialized boto3 client.
        """
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=self.config.region,
            aws_access_key_id=self.secrets.key,
            aws_secret_access_key=self.secrets.secret,
        )

    def generate_output(self) -> List[str]:
        """
        Generates output by sending the user input to the Bedrock model and processing the response.

        :return: List of output strings generated by the model.
        """
        user_inputs = self.content.user_input
        if len(user_inputs) == 1:
            output = self._generate_single_output(user_inputs[0])
        else:
            output = self._generate_multiple_outputs(user_inputs)
            output = ";;".join(output)
        return self._postprocess(output)

    def _generate_multiple_outputs(self, user_input: List[Dict[str, Any]]) -> List[str]:
        """
        Generates multiple outputs using concurrent futures for parallel processing.

        :param user_input: List of dictionaries representing batches of user input.
        :return: List of output strings generated by the model.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._generate_single_output, user_input): idx
                for idx, user_input in enumerate(user_input)
            }
            results = [""] * len(user_input)

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results

    def _generate_single_output(self, user_input: Dict[str, Any]) -> str:
        """
        Generates a single output by sending the user input to the Bedrock model.

        :param user_input: Dictionary representing a batch of user input.
        :return: Output string generated by the model.
        """
        try:
            user_message = self._create_user_message(user_input)
            temp_history = self.history + [user_message] if self.history is not None else [user_message]

            response = self._send_request(temp_history)
            response_data = self._parse_response(response)
            output = self._extract_output_from_response(response_data)

            if self.history is not None:
                self.history.append(user_message)
                self.history.append({"role": "assistant", "content": output})

            return output
        except Exception as e:
            raise Exception(f"Error generating output: {str(e)}")

    def _create_user_message(self, user_input: Dict[str, Any]) -> Dict[str, str]:
        """
        Creates a user message to send to the Bedrock model.

        :param user_input: Dictionary representing a batch of user input.
        :return: Dictionary representing the user message.
        """
        user_message_content = f"{self.config.prompt}\n{user_input}\nReturn no extra text."
        if self.config.batch_size != 1:
            user_message_content += "\nSeparate answers by a double semicolon (;;)."
        return {"role": "user", "content": user_message_content}

    def _send_request(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Sends a request to the Bedrock model with the specified messages.

        :param messages: List of dictionaries representing the messages to send.
        :return: Dictionary representing the model's response.
        """
        request_payload = json.dumps(
            {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": messages,
                "anthropic_version": ANTHROPIC_VERSION,
            }
        )

        return self.client.invoke_model(modelId=self.config.model, body=request_payload)

    @staticmethod
    def _parse_response(response: Dict) -> Dict:
        """
        Parses the response from the Bedrock model.

        :param response: Dictionary representing the model's response.
        :return: Dictionary containing the parsed response data.
        """
        response_body = response["body"].read()
        response_data = json.loads(response_body)
        if response_data.get("type") == "error":
            raise Exception(response_data["error"]["message"])
        return response_data

    @staticmethod
    def _extract_output_from_response(response_data: Dict) -> str:
        """
        Extracts the output from the model's response data.

        :param response_data: Dictionary containing the parsed response data.
        :return: Output string extracted from the response data.
        """
        if not response_data["content"]:
            raise Exception("Empty response from Claude")

        output = response_data["content"][0]["text"]
        if response_data.get("stop_reason") == "max_tokens":
            raise Exception(f"Output was cut off. Increase the 'max_tokens' amount. Response from Claude: {output}")
        return output

    @staticmethod
    def _postprocess(text: str) -> List[str]:
        """
        Post-processes the generated output text by splitting it into a list.

        :param text: String representing the raw output text.
        :return: List of strings representing the processed output.
        """
        return text.replace(";;;;", ";;").strip(";;").split(";;")


def format_validation_errors(errors) -> List[str]:
    """
    Formats validation errors from Pydantic models into a more readable format.

    :param errors: List of error dictionaries from Pydantic validation.
    :return: List of formatted error strings.
    """
    return [
        f"Error while validating: '{error['loc'][0]}' - {error['msg']}"
        for error in errors
    ]
