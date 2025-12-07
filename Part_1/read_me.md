Step 1 make a folder. Then open terminal and go to that folder

``` cd "/Users/steph/Desktop/World Model"```

Step 2. Create and activate a virtual environment

```python3 -m venv .venv

source .venv/bin/activate```

We should now see something like (.venv) at the start of the terminal prompt.


Step 3. Install the required packages

```pip install "gymnasium[classic-control]" minigrid numpy```

Step 4. Run the script

```python3 collect_minigrid_data.py```

