import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from opencompass.models.openai_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class WeLM(BaseAPIModel):
    """Model wrapper around WeLM's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        openai_proxy_url (str, optional): An optional proxy url to use when
            connecting to OpenAI's API. When set to 'ENV', the url will be
            fetched from the environment variable $OPENAI_PROXY_URL.
            Defaults to None.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
        tokenizer_path (str, optional): The path to the tokenizer. Use path if
            'tokenizer_path' is None, otherwise use the 'tokenizer_path'.
            Defaults to None.
        extra_body (Dict, optional): Add additional JSON properties to
            the request
        think_tag (str, optional): The tag to use for reasoning content.
            Defaults to '</think>'.
        max_workers (int, optional): Maximum number of worker threads for
            concurrent API requests. For I/O-intensive API calls, recommended
            value is 10-20. Defaults to None (uses CPU count * 2).
    """

    is_api: bool = True

    def __init__(
        self,
        path: str = "gpt-3.5-turbo",
        max_seq_len: int = 16384,
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        retry: int = 2,
        key: Union[str, List[str]] = "ENV",
        org: Optional[Union[str, List[str]]] = None,
        meta_template: Optional[Dict] = None,
        openai_api_base: str = None,
        openai_proxy_url: Optional[str] = None,
        mode: str = "none",
        logprobs: Optional[bool] = False,
        max_out_len: int = 1024,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tokenizer_path: Optional[str] = None,
        extra_body: Optional[Dict] = None,
        verbose: bool = False,
        think_tag: str = "</think>",
        max_workers: Optional[int] = None,
    ):

        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            retry=retry,
            verbose=verbose,
        )
        self.max_out_len = max_out_len
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p
        assert mode in ["none", "front", "mid", "rear"]
        self.mode = mode
        self.logprobs = logprobs
        self.logger.info(f"Start load hf tokenizer: {tokenizer_path}")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        self.extra_body = extra_body
        self.think_tag = think_tag

        if max_workers is None:
            cpu_count = os.cpu_count() or 1
            self.max_workers = min(32, (cpu_count + 5) * 2)
        else:
            self.max_workers = max_workers

        if isinstance(key, str):
            if key == "ENV":
                if "OPENAI_API_KEY" not in os.environ:
                    raise ValueError("OpenAI API key is not set.")
                self.keys = os.getenv("OPENAI_API_KEY").split(",")
            else:
                self.keys = [key]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = openai_api_base.split(",")

        if openai_proxy_url == "ENV":
            if "OPENAI_PROXY_URL" not in os.environ:
                raise ValueError("OPENAI_PROXY_URL is not set.")
            self.proxy_url = os.getenv("OPENAI_PROXY_URL")
        else:
            self.proxy_url = openai_proxy_url

        self.path = path

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 1024,
        stopping_criteria: List[str] = None,
        temperature: float = 0,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature
        if self.top_p is not None:
            top_p = self.top_p
        stop = []
        for words in [self.stop, stopping_criteria]:
            if words:
                stop += words
        stop = list(set(stop))
        if max_out_len is None:
            max_out_len = self.max_out_len
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._generate,
                        inputs,
                        [max_out_len] * len(inputs),
                        [stop] * len(inputs),
                        [temperature] * len(inputs),
                        [top_p] * len(inputs),
                    ),
                    total=len(inputs),
                    desc="Inferencing",
                )
            )
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int,
        stop: List[str],
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate results given a list of inputs.

        Args:
            input (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.
            top_p: An alternative to sampling with temperature, called nucleus sampling,
                where the model considers the results of the tokens with top_p probability mass.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptType))

        if isinstance(input, str):
            prompt = input
        else:  # Convert messages format into raw
            prompt = "\n\n".join([item["prompt"] for item in input])

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError("All keys have insufficient quota.")

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0

                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]

            header = {
                "Authorization": f"Bearer {key}",
                "content-type": "application/json",
                "api-key": key,
            }

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header["OpenAI-Organization"] = self.orgs[self.org_ctr]

            try:
                data = dict(
                    model=self.path,
                    prompt=prompt,
                    max_tokens=max_out_len,
                    n=1,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                )
                if self.extra_body:
                    data.update(self.extra_body)
                if isinstance(self.url, list):
                    import random

                    url = self.url[random.randint(0, len(self.url) - 1)]
                else:
                    url = self.url
                if self.proxy_url is None:
                    raw_response = requests.post(
                        url, headers=header, data=json.dumps(data)
                    )
                else:
                    proxies = {
                        "http": self.proxy_url,
                        "https": self.proxy_url,
                    }
                    if self.verbose:
                        self.logger.debug(f"Start send query to {self.proxy_url}")
                    raw_response = requests.post(
                        url,
                        headers=header,
                        data=json.dumps(data),
                        proxies=proxies,
                    )

                    if self.verbose:
                        self.logger.debug(f"Get response from {self.proxy_url}")

            except requests.ConnectionError:
                self.logger.error("Got connection error, retrying...")
                continue
            try:
                if raw_response.status_code != 200:
                    self.logger.error(
                        f"Request failed with status code "
                        f"{raw_response.status_code}, response: "
                        f"{raw_response.content.decode()}"
                    )
                    continue
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error(
                    f"JsonDecode error, got status code "
                    f"{raw_response.status_code}, response: "
                    f"{raw_response.content.decode()}"
                )
                continue
            self.logger.debug(str(response))
            try:
                # Extract content and reasoning_content from response
                content = response["choices"][0]["text"]
                return content
            except KeyError:
                if "error" in response:
                    if response["error"]["code"] == "rate_limit_exceeded":
                        time.sleep(10)
                        self.logger.warn("Rate limit exceeded, retrying...")
                        continue
                    elif response["error"]["code"] == "insufficient_quota":
                        self.invalid_keys.add(key)
                        self.logger.warn(f"insufficient_quota key: {key}")
                        continue
                    elif response["error"]["code"] == "invalid_prompt":
                        self.logger.warn("Invalid prompt:", str(input))
                        return ""
                    elif response["error"]["type"] == "invalid_prompt":
                        self.logger.warn("Invalid prompt:", str(input))
                        return ""

                    self.logger.error(
                        "Find error message in response: ",
                        str(response["error"]),
                    )
            max_num_retries += 1

        raise RuntimeError(
            "Calling OpenAI failed after retrying for "
            f"{max_num_retries} times. Check the logs for "
            "details."
        )

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.hf_tokenizer.encode(prompt))


@MODELS.register_module()
class WeLMRender(WeLM):
    def _generate(
        self,
        input: PromptType,
        max_out_len: int,
        stop: List[str],
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate results given a list of inputs.

        Args:
            input (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.
            top_p: An alternative to sampling with temperature, called nucleus sampling,
                where the model considers the results of the tokens with top_p probability mass.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptType))

        query = dict(max_out_len=max_out_len, stop=stop)
        return query