import supervisely as sly
from supervisely.app.widgets import Card, ClassesListSelector, Container, Field, TagsListSelector
import project_dataset.src.globals as g

select_classes = ClassesListSelector(multiple=True)
classes_field = Field(
    content=select_classes,
    title="Select classes",
    description="Select classes which should be kept after prediction, other classes will be ignored.",
)
select_tags = TagsListSelector(multiple=True)
tags_field = Field(
    content=select_tags,
    title="Select tags",
    description="Select tags which should be kept after prediction, other tags will be ignored.",
)

card = Card(
    "3️⃣ Choose classes and tags",
    "Choose classes and tags which should be kept after prediction.",
    content=Container([classes_field, tags_field]),
    collapsable=True,
    lock_message="Connect to the deployed neural network on step 2️⃣.",
)
card.lock()
card.collapse()


def load_classes() -> None:
    """Fills the widget with the classes from the model metadata."""
    obj_classes = g.model_meta.obj_classes
    select_classes.set(obj_classes)
    sly.logger.info(f"{len(obj_classes)} classes were loaded.")
    select_classes.select_all()


def load_tags():
    """Fills the widget with the tags from the model metadata."""
    obj_tags = g.model_meta.tag_metas
    select_tags.set(obj_tags)
    sly.logger.info(f"{len(obj_tags)} tags were loaded.")
    select_tags.select_all()
