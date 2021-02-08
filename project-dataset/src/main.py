import os
import supervisely_lib as sly

from globals import my_app
import init_ui as ui


def main():
    data = {}
    state = {}

    ui.init_input_project(data)

    my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)