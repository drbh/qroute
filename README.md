# qroute

> [!IMPORTANT]
> This is research code, use at your own risk.

This repo is a tiny exploration of using an online reinforcement learning algorithm to route requests to a set of language models.

This idea is inspired by the paper [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/pdf/2406.18665) and repo https://github.com/lm-sys/RouteLLM this repo implements a simple policy value network for online learning to route LLMs based on asynchronous votes from users.

## Setup

Start the trainable server/router with the following command:

```bash
python src/server.py
```

Now train the model to send requests about cats to one server and requests about dogs to another server. Running fot ~30 seconds should be enough to train the model.

```bash
python src/catdog.py
```

Now send a request to the server with the following command:

```bash
curl -X POST "http://localhost:7890/stream" -H "Content-Type: application/json" -d '{"input":"I love my cat"}'

# SERVER LOGS
# Model agi:             0.95%
# Model llama400b:       1.05%
# Model smart_model:     1.07%
# Model medium_model:    82.18%
# Model bad_model:       14.75%
# Selecting model based on policy
# Selected model: medium_model
```

great, the model has learned to route requests about cats to the `medium_model` model and we can try the same with dogs:

```bash
curl -X POST "http://localhost:7890/stream" -H "Content-Type: application/json" -d '{"input":"I love my dog"}'

# SERVER LOGS
# Model agi:             0.04%
# Model llama400b:       0.04%
# Model smart_model:     0.05%
# Model medium_model:    0.92%
# Model bad_model:       98.96%
# Selecting model based on policy
# Selected model: bad_model
```

The model has learned to route requests about dogs to the `bad_model` model.

💪
