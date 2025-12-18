# Dynamic Traffic Signal Control System

An intelligent traffic management system that leverages **Reinforcement Learning** (Q-Learning) to optimize traffic flow and reduce congestion within a simulated urban intersection.

## Prerequisites

- **Python** > 3.11
- **Poetry** for dependency management and environment isolation and Python 

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd "Dynamic Traffic Signal Control System"
    ```

2.  **Install dependencies and create the virtual environment:**
    ```bash
    poetry install
    ```

---

## Running Tests

To ensure the RL logic and environment simulations work correctly, run the following commands from the **root** directory (Dynamic Traffic Signal Control System) 

### Run all tests
This is the recommended command to verify the entire system:
```bash
poetry run pytest Tests/
```

### Run a specific test file
Execute one of the following command to test only 1 portion of the system:
```bash
poetry run python Tests/test_environment.py
poetry run python Tests/test_QLearn.py
poetry run python Tests/test_training_running.py

```
## Run the Agent and the simulation
Execute :
```bash
poetry run python main.py -e 10 -r
```