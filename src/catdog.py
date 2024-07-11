import requests
import random
import time

API_BASE_URL = "http://localhost:7890"
STREAM_ENDPOINT = f"{API_BASE_URL}/stream"
VOTE_ENDPOINT = f"{API_BASE_URL}/vote"


def send_request(input_text):
    """Send a request to the stream endpoint and return the response."""
    response = requests.post(STREAM_ENDPOINT, json={"input": input_text})
    if response.status_code == 200:
        # extract which model was selected "Data from X Model"
        response_model = (
            response.text.split("Data from ")[1].split(" Model")[0].strip().lower()
        )
        # extract the request ID
        request_id = response.text.split("Request ID:")[1].split("\n")[0].strip()
        return request_id, response_model
    else:
        print(f"Error sending request: {response.status_code}")
        return None


def vote(request_id, vote_value):
    """Send a vote for a specific request."""
    response = requests.post(
        VOTE_ENDPOINT, json={"request_id": request_id, "vote": vote_value}
    )
    if response.status_code == 200:
        print(f"Vote sent successfully for request {request_id}")
    else:
        print(f"Error sending vote: {response.status_code}")


def train_model():
    """Train the model by sending requests and voting."""
    cat_inputs = [
        "I absolutely adore my cat",
        "Cats make wonderful companions",
        "My cat is incredibly playful",
        "Cats spend a lot of time sleeping",
        "The cat meowed very loudly",
        "My cat loves chasing laser pointers",
        "Cats have such independent personalities",
        "The cat purrs when I pet it",
        "My cat enjoys lounging in sunny spots",
        "Cats are very curious creatures",
        "The cat loves to climb on furniture",
        "My cat often hides in small spaces",
        "Cats are great for apartment living",
        "The cat has a very soft fur coat",
        "My cat likes to watch birds from the window",
    ]

    dog_inputs = [
        "I take my dog for a walk every day",
        "Dogs are truly man's best friend",
        "My dog absolutely loves playing fetch",
        "Dogs require regular exercise",
        "The dog barked at the mail carrier",
        "My dog enjoys swimming in the lake",
        "Dogs are very loyal animals",
        "The dog wags its tail when it's happy",
        "My dog loves to chew on bones",
        "Dogs can be trained to do many tricks",
        "The dog loves to play in the park",
        "My dog enjoys car rides",
        "Dogs are great for security",
        "The dog is very friendly with children",
        "My dog always greets me excitedly when I come home",
    ]

    other_inputs = [
        "I keep a pet fish in my aquarium",
        "Birds are chirping cheerfully outside",
        "Rabbits also make fantastic pets",
        "I have a preference for keeping snakes as pets",
        "My hamster enjoys running on its wheel",
        "Fish are very calming to watch",
        "Birds can be very colorful and lively",
        "Rabbits have very soft fur",
        "Snakes are fascinating to observe",
        "Hamsters are very active at night",
        "I have a pet turtle that loves basking under a heat lamp",
        "Birds can be trained to mimic sounds",
        "Rabbits enjoy hopping around the garden",
        "Snakes need a warm environment to thrive",
        "Hamsters store food in their cheeks",
    ]

    for _ in range(5000):
        if random.choice([True, False]):
            input_text = random.choice(cat_inputs)
            expected_model = "medium"
        else:
            input_text = random.choice(dog_inputs)
            expected_model = "bad"

        print(f"\nSending request for: {input_text}")
        print(f"Expecting model: {expected_model}")

        request_id, response_model = send_request(input_text)
        print(f"Received response from: {response_model}")
        if request_id:
            time.sleep(0.001)

            if "cat" in input_text.lower() and expected_model == response_model:
                print("Sending upvote...")
                vote(request_id, 1)  # upvote
            elif "dog" in input_text.lower() and expected_model == response_model:
                print("Sending downvote...")
                vote(request_id, 1)  # upvote
            else:
                print("Sending downvote...")
                vote(request_id, 0)  # downvote (no effect)
                pass


if __name__ == "__main__":
    # elegant exit if ctrl+c is pressed
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        exit(0)
