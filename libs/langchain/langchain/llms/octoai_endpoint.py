from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM, create_base_retry_decorator
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, root_validator
from langchain.utils import get_from_dict_or_env
from langchain.schema.output import GenerationChunk


def _stream_response_to_generation_chunk(
    stream_response: Any,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    return GenerationChunk(
        text=stream_response.choices[0].text,
        generation_info=dict(
            finish_reason=stream_response.choices[0].finish_reason,
            logprobs=stream_response.choices[0].logprobs,
        ),
    )


class OctoAIEndpoint(LLM):
    """OctoAI LLM Endpoints.

    OctoAIEndpoint is a class to interact with OctoAI
     Compute Service large language model endpoints.

    To use, you should have the ``octoai-sdk`` python package installed, and the
    environment variable ``OCTOAI_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.llms.octoai_endpoint  import OctoAIEndpoint
            OctoAIEndpoint(
                octoai_api_token="octoai-api-key",
                endpoint_url="https://mpt-7b-demo-kk0powt97tmb.octoai.cloud/generate",
                model_kwargs={
                    "max_new_tokens": 200,
                    "temperature": 0.75,
                    "top_p": 0.95,
                    "repetition_penalty": 1,
                    "seed": None,
                    "stop": [],
                },
            )

    """

    endpoint_url: Optional[str] = None
    """Endpoint URL to use."""

    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    octoai_api_token: Optional[str] = None
    """OCTOAI API Token"""

    max_retries: int = 20
    """Max retries in case of failures"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        octoai_api_token = get_from_dict_or_env(
            values, "octoai_api_token", "OCTOAI_API_TOKEN"
        )
        values["endpoint_url"] = get_from_dict_or_env(
            values, "endpoint_url", "ENDPOINT_URL"
        )

        values["octoai_api_token"] = octoai_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "octoai_endpoint"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs.update(kwargs)

        # Prepare the payload JSON
        params: dict = {
            "endpoint_url": self.endpoint_url,
            "inputs": {"inputs": prompt, "parameters": _model_kwargs},
        }

        response = completion_with_retry(
            self, run_manager=run_manager, stop=stop, **params
        )

        return response["generated_text"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs.update(kwargs)

        # Prepare the payload JSON
        params: dict = {
            "endpoint_url": self.endpoint_url,
            "inputs": {"inputs": prompt, "parameters": _model_kwargs},
        }
        for stream_resp in completion_with_retry_streaming(
            self, run_manager=run_manager, stop=stop, **params
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            yield chunk


def completion_with_retry(
    llm: OctoAIEndpoint,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Use tenacity to retry the completion call."""
    from octoai import client

    octoai_client = client.Client(token=llm.octoai_api_token)
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Mapping[str, Any]:
        return octoai_client.infer(**kwargs)

    return _completion_with_retry(**kwargs)


def completion_with_retry_streaming(
    llm: OctoAIEndpoint,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Iterator[dict]:
    """Use tenacity to retry the completion call for streaming."""
    from octoai import client

    octoai_client = client.Client(token=llm.octoai_api_token)
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Iterator[dict]:
        return octoai_client.infer_stream(**kwargs)

    return _completion_with_retry(**kwargs)


def _create_retry_decorator(
    llm: OctoAIEndpoint,
    *,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Define retry mechanism."""
    from octoai import client

    errors = [client.error.OctoAIClientError]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
