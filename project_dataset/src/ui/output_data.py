from supervisely.app.widgets import Button, Card, Container, Input

import project_dataset.src.globals as g

output_project_name = Input(f"{g.project_info.name}_inference", minlength=1)
apply_button = Button("Apply model to input data", icon="zmdi zmdi-check")

card = Card(
    "6️⃣ Output data",
    "New project with predictions will be created. Original project will not be modified.",
    content=Container([output_project_name, apply_button]),
    collapsable=True,
    lock_message="Connect to the deployed neural network on step 2️⃣.",
)
card.lock()
card.collapse()
