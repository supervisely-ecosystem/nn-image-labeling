<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/106374579/187226641-dce1a180-66b8-45ad-840e-ab5e445ee342.png"/>

# NN Image Labeling

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Use</a> •
  <a href="#Demo">Demo</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../../supervisely-ecosystem/nn-image-labeling/annotation-tool)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/nn-image-labeling)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/nn-image-labeling/annotation-tool.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/nn-image-labeling/annotation-tool.png)](https://supervisely.com)

</div>

# Overview

Any NN can be integrated into Labeling interface if it has properly implemented serving app (for example: Serve YOLOv5). App adds classes and tags to project automatically.

<a data-key="sly-embeded-video-link" href="https://youtu.be/eWAvbmkm6JQ" data-video-code="eWAvbmkm6JQ">
    <img src="https://i.imgur.com/ODlVoBh.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

# What's New

### v1.7.5

- NN Models can now be applied to `Polyline` figures. All points of the polyline are positive points

<div align="center" markdown>
  <img src="https://github.com/user-attachments/assets/3b26f64e-2ab0-4a24-9f04-a5b9ff084d4b" width="50%">
</div>

### v1.6.0

- ☑️ Added checkbox to always add suffix to model class/tag name even if it doesn't have conflicts with existing one

<div align="center" markdown>
  <img src="https://user-images.githubusercontent.com/48913536/235165230-f5c5cab6-8076-4c4d-929f-16ceefe6894f.png" width="50%">
</div>

# How To Use

0. Add this app to your team from Ecosystem
1. Be sure that NN you are going to use is deployed in your team
2. To start using app, user has to run it (from Team Apps page or directly in labeling UI) or open already running session. App doesn't support multiuser mode: it means that every user has to run its own session, BUT multiple sessions can connect to a single NN. 
   
    For example: There are 5 labelers in your team and you would like to use YOLOv5. In that case you should have at least one session of the deployed NN and run separate sessions of this app for every user.
    
3. Apply model to image area defined by object bbox. If user selects object of interest, app creates bounding box around object and applies model to this image area (ROI).

# Demo

|                                                       Full Image                                                        |                                                         ROI                                                          |
| :---------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/supervisely-ecosystem/nn-image-labeling/releases/download/v1.0.0/det_full_image-min.gif"/> | <img src="https://github.com/supervisely-ecosystem/nn-image-labeling/releases/download/v1.0.0/det_obj_roi-min.gif"/> |
| <img src="https://github.com/supervisely-ecosystem/nn-image-labeling/releases/download/v1.0.0/seg_full_image-min.gif"/> | <img src="https://github.com/supervisely-ecosystem/nn-image-labeling/releases/download/v1.0.0/seg_obj_roi-min.gif"/> |







