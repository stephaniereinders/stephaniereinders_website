+++
title = "Algorithm Mismatch in Spatial Steganalysis"

# Date first published.
date = "2019-02-01"

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["Stephanie Reinders", "Li Lin", "Yong Guan", "Min Wu", "Jennifer Newman"]

# Publication type.
# Legend:
# 0 = Uncategorized
# 1 = Conference proceedings
# 2 = Journal
# 3 = Work in progress
# 4 = Technical report
# 5 = Book
# 6 = Book chapter
publication_types = ["1"]

# Publication name and optional abbreviated version.
publication = "In *Electronic Imaging*"
publication_short = "In *EI*"

# Abstract and optional shortened version.
abstract = "The number and availability of stegonographic embedding algorithms continues to grow. Many traditional blind steganalysis frameworks require training examples from every embedding algorithm, but collecting, storing and processing representative examples of each algorithm can quickly become untenable. Our motivation for this paper is to create a straight-forward, non- data-intensive framework for blind steganalysis that only requires examples of cover images and a single embedding algorithm for training. Our blind steganalysis framework addresses the case of algorithm mismatch, where a classifier is trained on one algorithm and tested on another, with four spatial embedding algorithms: LSB matching, MiPOD, S-UNIWARD and WOW.
We use RAW image data from the BOSSbase database and and data collected from six iPhone devices. Ensemble Classifiers with Spatial Rich Model features are trained on a single embedding algorithm and tested on each of the four algorithms. Classifiers trained on MiPOD, S-UNIWARD and WOW data achieve decent error rates when testing on all four algorithms. Most notably, an Ensemble Classifier with an adjusted decision threshold trained on LSB matching data achieves decent detection results on MiPOD, S-UNIWARD and WOW data."
abstract_short = "We present a novel blind steganalysis framework that does not require large amounts of training data."

# Featured image thumbnail (optional)
image_preview = ""

# Is this a selected publication? (true/false)
selected = true

# Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter the filename (excluding '.md') of your project file in `content/project/`.
#   E.g. `projects = ["deep-learning"]` references `content/project/deep-learning.md`.
projects = []

# Links (optional).
url_pdf = "https://www.ingentaconnect.com/contentone/ist/ei/2019/00002019/00000005/art00010?crawler=true&mimetype=application/pdf"
url_preprint = ""
url_code = ""
url_dataset = ""
url_project = ""
url_slides = ""
url_video = ""
url_poster = ""
url_source = ""

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
# url_custom = [{name = "Custom Link", url = "http://example.org"}]

# Does the content use math formatting?
math = true

# Does the content use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = ""
caption = "steganalysis"

+++
