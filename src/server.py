import asyncio
from uuid import uuid4
from collections import deque
import random
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn

from modeling import PolicyValueNetwork
from modeling import (
    STATE_DIM,
    NUM_ACTIONS,
    LEARNING_RATE,
    VALUE_WEIGHT,
    ENTROPY_WEIGHT,
)

BATCH_SIZE = 8
MAX_MEMORY_SIZE = 100  # 00

policy_value_net = PolicyValueNetwork(STATE_DIM, NUM_ACTIONS)
optimizer = optim.Adam(policy_value_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MAX_MEMORY_SIZE)


class InputData(BaseModel):
    input: str


class VoteData(BaseModel):
    request_id: str
    vote: int  # 1 for upvote, -1 for downvote


NUM_MSGS = 5


async def agi():
    for i in range(NUM_MSGS):
        yield f"Data from AGI\n"
        await asyncio.sleep(0.025)


async def llama400b():
    for i in range(NUM_MSGS):
        yield f"Data from LLaMA 400B\n"
        await asyncio.sleep(0.025)


async def smart_model():
    for i in range(NUM_MSGS):
        yield f"Data from Smart Model\n"
        await asyncio.sleep(0.025)


async def medium_model():
    for i in range(NUM_MSGS):
        yield f"Data from Medium Model\n"
        await asyncio.sleep(0.025)


async def bad_model():
    for i in range(NUM_MSGS):
        yield f"Data from Bad Model\n"
        await asyncio.sleep(0.025)


available_models = [agi, llama400b, smart_model, medium_model, bad_model]


def select_action(state):
    with torch.no_grad():
        policy_logits, _ = policy_value_net(state)
        action_probs = torch.softmax(policy_logits, dim=-1)
        for i, prob in enumerate(action_probs.squeeze().tolist()):
            fname = available_models[i].__name__
            fname = fname + ":" + " " * (15 - len(fname))
            print(f"Model {fname} {round(prob * 100, 2)}%")

    # one 1/3 of the time, choose randomly
    if random.random() < 0.33:
        print("Randomly selecting model")
        choice = random.randint(0, 4)
    else:
        print("Selecting model based on policy")
        choice = torch.argmax(action_probs).item()

    print(f"Selected model: {available_models[choice].__name__}")
    return choice


app = FastAPI()


@app.post("/stream")
async def stream_route(input_data: InputData, background_tasks: BackgroundTasks):
    state = policy_value_net.get_state_embedding(input_data.input)
    action = select_action(state)
    request_id = str(uuid4())

    async def event_stream():
        yield f"Request ID: {request_id}\n"
        async for data in available_models[action]():
            yield data

    # store the experience without reward
    background_tasks.add_task(lambda: memory.append((state, action, None, request_id)))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/vote")
async def vote(vote_data: VoteData, background_tasks: BackgroundTasks):
    for i, (state, action, _, request_id) in enumerate(memory):
        if request_id == vote_data.request_id:
            memory[i] = (state, action, vote_data.vote, request_id)
            break

    # batch train if we have enough data
    if len(memory) >= BATCH_SIZE:
        background_tasks.add_task(train_batch)

    return {"status": "Vote recorded"}


async def train_batch():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, _ = zip(*batch)

    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(
        [r if r is not None else 0 for r in rewards], dtype=torch.float32
    )

    policy_logits, values = policy_value_net(states)
    action_probs = torch.softmax(policy_logits, dim=-1)

    selected_action_probs = action_probs[torch.arange(BATCH_SIZE), actions]
    log_probs = torch.log(selected_action_probs)

    policy_loss = -(log_probs * rewards).mean()
    value_loss = nn.MSELoss()(values.squeeze(), rewards)
    entropy = -(action_probs * torch.log(action_probs)).sum(dim=1).mean()

    total_loss = policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Batch trained. Loss: {total_loss.item():.4f}")


@app.get("/stats")
async def get_stats():
    return {
        "model_parameters": sum(p.numel() for p in policy_value_net.parameters()),
        "learning_rate": LEARNING_RATE,
        "entropy_weight": ENTROPY_WEIGHT,
        "value_weight": VALUE_WEIGHT,
        "batch_size": BATCH_SIZE,
        "memory_size": len(memory),
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7890)
