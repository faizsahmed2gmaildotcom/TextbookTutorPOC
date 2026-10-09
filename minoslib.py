from typing import Callable


def validateInput(validation: Callable[[str], bool], user_prompt: str):
    user_input = input(user_prompt)
    while not validation(user_input):
        user_input = input(user_prompt)

    return user_input
