<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/106374579/187227957-ea4fd452-35ef-4969-9e55-cd7a5a4873ee.png"/>


# Apply NN to Images Project


<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#whats-new">What's New</a> •
  <a href="#common-apps">Common Apps</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../../supervisely-ecosystem/nn-image-labeling/project-dataset)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/nn-image-labeling)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/nn-image-labeling/project-dataset.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/nn-image-labeling/project-dataset.png)](https://supervisely.com)

</div>

# Overview

App connects to the deployed Neural Network and applies it to the images project. It allows to configure inference settings, model output classes and tags, and preview predictions. App adds classes and tags to project automatically.

<a data-key="sly-embeded-video-link" href="https://youtu.be/DUQgr_SLVR4" data-video-code="DUQgr_SLVR4">
    <img src="https://i.imgur.com/Edy7B1H.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:50%;">
</a>

# What's New

### v1.7.41

- Added support for saving and loading inference settings from `.yaml` template file. Now you can save your settings templates and load them later to apply to the same or another project. Settings are saved to the template folder located in the Team Files: `nn-image-labeling/inf-settings-templates/`.

![save-load-inf-settings](https://github.com/user-attachments/assets/ff5c68ac-ed0f-41f0-966f-d61c0b6b8d25)

### v1.6.0

- ☑️ Added checkbox to always add suffix to model class/tag name even if it doesn't have conflicts with existing one

<div align="center" markdown>
  <img src="https://user-images.githubusercontent.com/48913536/235165358-8683c97d-05fb-437a-a386-09eff3e1203a.png" width="50%">
</div>

### v1.2.3

- Configurable sliding window mode with preview, mode enables automatically if the connected neural network supports it

https://user-images.githubusercontent.com/33942379/171401791-f389eee4-4fdc-4fed-bbcc-409dc306e045.mp4

- Inference preview ob both random or specific image

https://user-images.githubusercontent.com/33942379/171401798-0b425683-4138-4b01-b008-50f0fc98eb0e.mp4

# How To Run

0. Add this app to your team from Ecosystem
1. Be sure that NN you are going to use is deployed in your team
2. Start app from the context menu of the images project or dataset. 
