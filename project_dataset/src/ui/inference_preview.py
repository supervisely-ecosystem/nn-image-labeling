import supervisely as sly
from supervisely.app.widgets import Button, Card, Checkbox, Flexbox, Grid, LabeledImage

original_preview = LabeledImage()
inference_preview = LabeledImage()
preview_grid = Grid([original_preview, inference_preview], columns=2)
random_image_checkbox = Checkbox("Random image", checked=True)
preview_button = Button("Preview", icon="zmdi zmdi-eye")

card = Card(
    "5️⃣ Inference preview",
    "Preview the model inference on a random image.",
    content=preview_grid,
    content_top_right=Flexbox([random_image_checkbox, preview_button]),
    collapsable=True,
    lock_message="Connect to the deployed neural network on step 2️⃣.",
)
card.lock()
card.collapse()
