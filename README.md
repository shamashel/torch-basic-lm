# Basic Torch Language Model

Based on the [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) video from Andrej Karpathy.

This is just an attempt to recreate the transformer Andrej made in his video with the goal of learning more about torch, transformers, and neural networks in general.

To run, make sure `python` `3.10` and `poetry` are installed. You can then run `poetry install` to get the dependencies (it's just torch and numpy).

Finally, you can run the code with `poetry run python ./main.py`

Note that the first run will train the model and then save the trained weights to `model.pth`. Subsequent runs will load these weights.