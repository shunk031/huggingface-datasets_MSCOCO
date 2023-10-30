---
annotations_creators:
- crowdsourced
language:
- en
language_creators:
- found
license:
- cc-by-4.0
multilinguality:
- monolingual
pretty_name: MSCOCO
size_categories: []
source_datasets:
- original
tags:
- image-captioning
- object-detection
- keypoint-detection
- stuff-segmentation
- panoptic-segmentation
task_categories:
- image-segmentation
- object-detection
- other
task_ids:
- instance-segmentation
- semantic-segmentation
- panoptic-segmentation
---

# Dataset Card for MSCOCO

[![CI](https://github.com/shunk031/huggingface-datasets_MSCOCO/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_MSCOCO/actions/workflows/ci.yaml)

## Table of Contents
- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://cocodataset.org/#home
- **Repository:** https://github.com/shunk031/huggingface-datasets_MSCOCO
- **Paper (Preprint):** https://arxiv.org/abs/1405.0312
- **Paper (ECCV2014):** https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48
- **Leaderboard (Detection):** https://cocodataset.org/#detection-leaderboard
- **Leaderboard (Keypoint):** https://cocodataset.org/#keypoints-leaderboard
- **Leaderboard (Stuff):** https://cocodataset.org/#stuff-leaderboard
- **Leaderboard (Panoptic):** https://cocodataset.org/#panoptic-leaderboard
- **Leaderboard (Captioning):** https://cocodataset.org/#captions-leaderboard
- **Point of Contact:** info@cocodataset.org

### Dataset Summary

> COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features:
> - Object segmentation
> - Recognition in context
> - Superpixel stuff segmentation
> - 330K images (>200K labeled)
> - 1.5 million object instances
> - 80 object categories
> - 91 stuff categories
> - 5 captions per image
> - 250,000 people with keypoints

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

[More Information Needed]

## Dataset Structure

### Data Instances

#### 2014

- captioning dataset

```python
import datasets as ds

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2014,
    coco_task="captions",
)
```

- instances dataset

```python
import datasets as ds

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2014,
    coco_task="instances",
    decode_rle=True, # True if Run-length Encoding (RLE) is to be decoded and converted to binary mask.
)
```

- person keypoints dataset

```python
import datasets as ds

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2014,
    coco_task="person_keypoints",
    decode_rle=True, # True if Run-length Encoding (RLE) is to be decoded and converted to binary mask.
)
```

#### 2017

- captioning dataset

```python
import datasets as ds

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2017,
    coco_task="captions",
)
```

- instances dataset

```python
import datasets as ds

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2017,
    coco_task="instances",
    decode_rle=True, # True if Run-length Encoding (RLE) is to be decoded and converted to binary mask.
)
```

- person keypoints dataset

```python
import datasets as ds

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2017,
    coco_task="person_keypoints",
    decode_rle=True, # True if Run-length Encoding (RLE) is to be decoded and converted to binary mask.
)
```

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

[More Information Needed]

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

[More Information Needed]

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

> The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).
>
> ## Images
> The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use. The users of the images accept full responsibility for the use of the dataset, including but not limited to the use of any copies of copyrighted images that they may create from the dataset.
>
> ## Software
> Copyright (c) 2015, COCO Consortium. All rights reserved. Redistribution and use software in source and binary form, with or without modification, are permitted provided that the following conditions are met:
> - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
> - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
> - Neither the name of the COCO Consortium nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
> 
> THIS SOFTWARE AND ANNOTATIONS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### Citation Information

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

### Contributions

Thanks to [COCO Consortium](https://cocodataset.org/#people) for creating this dataset.
