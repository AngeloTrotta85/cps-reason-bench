# Providers

The adapters shipped here are the existing native Ollama and OpenAI-compatible
Chat Completions implementations. They use Python's standard-library HTTP client.
There is no direct Anthropic Messages adapter, Responses-only adapter, browser
automation, or provider SDK dependency in this release.

## Local Ollama

Install Ollama using its [official setup guide](https://docs.ollama.com/quickstart).
Ollama serves model inference locally. Start the service if it is not already
running, inspect installed models, and explicitly download the chosen model:

```sh
ollama serve
# In another terminal:
ollama ls
ollama pull qwen3.5:0.8b
cps-reason run configs/examples/local_quickstart.yaml
```

`serve`, `ls` and `pull` are documented in the
[Ollama CLI reference](https://docs.ollama.com/cli). The adapter sends non-streaming
requests to [`/api/chat`](https://docs.ollama.com/api/chat), with `think`, optional
format constraints, and decoding options. This release accepts a boolean `think`;
it does not expose every option newer servers might support. Model support and
memory consumption must be checked on your machine. Neither validation nor tests
start the service, pull weights, or infer from a real model.

`backend_parameters.structured_output_mode` defaults to `json_schema` in the
existing adapter. The provided Graph-v2 examples explicitly select `prompt_only`.
Switching it changes the model-interface condition. `num_ctx` can increase RAM use;
start with one model and one episode. There is no parallel worker option here.
Native Ollama has no credential transport in this wrapper; do not expose an
unauthenticated Ollama endpoint publicly.

## Compatible Cloud Endpoints

Set `backend: openai-compatible`, the actual `model` identifier, API `base_url`,
and `api_key_env`. For OpenRouter use `provider: openrouter`; other compatible
services use `custom` unless the existing `openai` behavior is intended. The
adapter appends `/chat/completions` unless already present.

```yaml
name: my_cloud_configuration
backend: openai-compatible
model: your-service-model-id
base_url: https://your-service.example/v1
api_key_env: MY_PROVIDER_KEY
backend_parameters:
  provider: custom
  structured_output_mode: prompt_only
  transport_retry: {max_attempts: 1}
```

The domain and model above are explicitly placeholders. The runnable OpenRouter
example names a model used in the research environment, but its current route,
availability, pricing and parameter support have not been live-qualified by this
release task. Review those with your provider before paying for calls.

The API key is read only from the named environment variable. Never include a key
in `base_url`, YAML, command arguments or committed `.env`. `.env.example` is a
template, not an automatically loaded credential store. Interactive shell entry:

```sh
read -rs OPENROUTER_API_KEY
export OPENROUTER_API_KEY
printf '\n'
cps-reason validate configs/examples/cloud_quickstart.yaml
cps-reason dry-run configs/examples/cloud_quickstart.yaml
cps-reason run configs/examples/cloud_quickstart.yaml
unset OPENROUTER_API_KEY
```

Use account-side spend limits as well as `runtime.max_episodes`. The release CLI
does not enforce a dollar budget. One episode can make several inference calls;
tokens and timeouts affect cost. `--yes` authorizes cloud execution, not a promise
of successful completion. All compatible endpoints require authorization, even a
loopback service, because locality alone does not establish cost.

## Routing, Errors and Normalization

OpenRouter `benchmark` routing forbids fallback and requires requested parameter
support. Optional `only`/`provider_order`, privacy and returned-identity constraints
are explicit controls; no route or privacy guarantee is inferred from a model
name. Unsupported settings can cause an infrastructure failure before reasoning
is evaluated. See the [configuration reference](CONFIGURATION.md).

Transport errors, rate limits, invalid provider envelopes, timeouts and empty
responses are not evidence of task-reasoning failure. The runner records the
classification and stops on infrastructure failure. Only explicitly configured,
bounded transport retries can occur, and ambiguous calls are not automatically
retried. A real provider response with invalid decision JSON is instead a protocol
outcome within an episode.

`markdown_single_fence` accepts only an optional single strict outer Markdown
fence and leaves payload meaning unchanged. Outside prose, multiple fences and
invalid JSON are not repaired. Strict and normalized runs are separate interface
conditions; never silently rescore one as the other.
