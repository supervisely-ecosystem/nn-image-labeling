<div align="center" markdown>
<img src="https://i.imgur.com/AFv8KQa.png"/>

# NN Image Labeling

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Use</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/nn-image-labeling/annotation-tool)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/nn-image-labeling)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/nn-image-labeling/annotation-tool&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/nn-image-labeling/annotation-tool&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/nn-image-labeling/annotation-tool&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Any NN can be integrated into Labeling interface if it has properly implemented serving app (for example: Serve YOLOv5). App adds classes and tags to project automatically.

<a data-key="sly-embeded-video-link" href="https://youtu.be/eWAvbmkm6JQ" data-video-code="eWAvbmkm6JQ">
    <img src="https://i.imgur.com/ODlVoBh.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

# How To Use

0. Add this app to your team from Ecosystem
1. Be sure that NN you are going to use is deployed in your team
2. To start using app, user has to run it (from Team Apps page or directly in labeling UI) or open already running session. App doesn't support multiuser mode: it means that every user has to run its own session, BUT multiple sessions can connect to a single NN. 
   
    For example: There are 5 labelers in your team and you would like to use YOLOv5. In that case you should have at least one session of the deployed NN and run separate sessions for every user.
3. To apply model to single object simply select this object and press `Apply model to ROI`

<p>
  <img src="https://github.com/supervisely-ecosystem/nn-image-labeling/releases/download/v1.0.0/apply-to-object-highlighted-min.gif" style="width:80%;"/>
</p>  





