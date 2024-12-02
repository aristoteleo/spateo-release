<p align="center">
  <img height="150" src="https://raw.githubusercontent.com/aristoteleo/spateo-release/main/docs/_static/logo.png" />
</p

[![documentation](https://readthedocs.org/projects/spateo-release/badge/?version=latest)](https://spateo-release.readthedocs.io/en/latest/)

[Installation](https://spateo-release.readthedocs.io/en/latest/installation.html) - [Tutorials](https://spateo-release.readthedocs.io/en/latest/tutorials/index.html) - [API](https://spateo-release.readthedocs.io/en/latest/autoapi/spateo/index.html) - [Citation](https://www.cell.com/cell/fulltext/S0092-8674(24)01159-0) - [Technical](https://spateo-release.readthedocs.io/en/latest/technicals/index.html)

![Spateo](https://github.com/user-attachments/assets/9581284c-0617-4561-8827-81134618dabf)

# Citation
### <b> Spatiotemporal modeling of molecular holograms </b>

Xiaojie Qiu<sup>1, 7, 8, $, </sup>\*, Daniel Y. Zhu<sup>3, $</sup>, Yifan Lu<sup>1, 7, 8, 9, $</sup>, Jiajun Yao<sup>2, 4, 10, $</sup>, Zehua Jing<sup>2, 4, 11, $</sup>, Kyung Hoi (Joseph) Min<sup>12, $</sup>, Mengnan Cheng<sup>2, 6, $</sup>, Hailin Pan<sup>6</sup>, Lulu Zuo<sup>6</sup>, Samuel King<sup>13</sup>, Qi Fang<sup>2, 6</sup>, Huiwen Zheng<sup>2, 11</sup>, Mingyue Wang<sup>2, 14</sup>, Shuai Wang<sup>2, 11</sup>, Qingquan Zhang<sup>25</sup>, Sichao Yu<sup>5</sup>, Sha Liao<sup>6, 17, 18</sup>, Chao Liu<sup>15</sup>, Xinchao Wu<sup>2, 4, 16</sup>, Yiwei Lai<sup>6</sup>, Shijie Hao<sup>2</sup>, Zhewei Zhang<sup>2, 4, 16</sup>, Liang Wu<sup>18</sup>, Yong Zhang<sup>15</sup>, Mei Li<sup>17</sup>, Zhencheng Tu<sup>2, 11</sup>, Jinpei Lin<sup>2, 4</sup>, Zhuoxuan Yang<sup>2, 16</sup>, Yuxiang Li<sup>15</sup>, Ying Gu<sup>2, 6, 11</sup>, Ao Chen<sup>6, 17, 18</sup>, Longqi Liu<sup>2, 19, 20</sup>, Jonathan S. Weissman<sup>5, 22, 23</sup>, Jiayi Ma<sup>9, </sup>\*, Xun Xu<sup>2, 11, 21, </sup>\*, Shiping Liu<sup>2, 19, 20, 24, </sup>\*, Yinqi Bai<sup>4, 26, </sup>\*  

<sup>$</sup> Co-first authors

\* Corresponding authors
 
https://www.cell.com/cell/fulltext/S0092-8674(24)01159-0 

# Abstract

<p align="justify">
Quantifying spatiotemporal dynamics during embryogenesis is crucial for understanding congenital diseases. We developed Spateo (https://github.com/aristoteleo/spateo-release), a 3D spatiotemporal modeling framework, and applied it to a 3D mouse embryogenesis atlas at E9.5 and E11.5, capturing eight million cells. Spateo enables scalable, partial, non-rigid alignment, multi-slice refinement, and mesh correction to create molecular holograms of whole embryos. It introduces digitization methods to uncover multi-level biology from subcellular to whole organ, identifying expression gradients along orthogonal axes of emergent 3D structures, e.g., secondary organizers such as midbrain-hindbrain boundary (MHB). Spateo further jointly models intercellular and intracellular interaction to dissect signaling landscapes in 3D structures, including the zona limitans intrathalamica (ZLI). Lastly, Spateo introduces “morphometric vector fields” of cell migration and integrates spatial differential geometry to unveil molecular programs underlying asymmetrical murine heart organogenesis and others, bridging macroscopic changes with molecular dynamics. Thus, Spateo enables the study of organ ecology at a molecular level in 3D space over time.
</p>

# Keywords

<p align="justify">
Spateo, whole embryo 3D spatial transcriptomics, 3D reconstruction, Stereo-seq, spatial domain digitization, ligand-receptor cell-cell interactions, intercellular and intracellular interactions, organogenesis mode, morphometric vector field, spatial differential geometry analyses
</p>


## Highlights of Spateo:

*  <p align="justify"> Spateo introduces a sophisticated approach, Starro, to segment single cells based purely on RNA signal, unsupervisedly identify continuous tissue domains via spatially-constrained clustering, and dissect the intricate spatial cell type distribution and tissue composition;

* <p align="justify"> Spateo identifies spatial polarity/gradient genes (e.g. neuronal layer-specific genes) by solving a partial differential equation to digitize layers and columns of a spatial domain. </p

* <p align="justify"> Spateo implements a full suite of spatially-aware modules for differential expression inference, including novel parametric models for spatially-informed prediction of cell-cell interactions and interpretable estimation of downstream effects. </p

* <p align="justify"> Spateo enables the reconstruction of 3D whole-organ models from 2D slices, identifying different “organogenesis modes” (patterns of cell migration during organogenesis) for each organ and quantifying morphometric properties (such as organ surface area, volume, length and cell density) over time. </p

* <p align="justify"> Spateo brings in the concept of the “morphometric vector field” that predicts migration paths for each cell within an organ in a 3D fashion and reveals principles of cell migration by exploring various differential geometry quantities. </p

## News
* Nov/11/2024: We are also honored to have this work highlighted by Nature: https://nature.com/articles/d41586-024-03615-8.  
* Nov/11/2024: We are thrilled to share the publication of Spateo in Cell today: https://cell.com/cell/fulltext/S0092-8674(24)01159-0. 


## Spateo Development Process
- Follow feature-staging-main review process
    - create a specific branch for new features
    - implement and test on your branch; add unit tests
    - create pull request
    - discuss with lab members and merge into the main branch once all checks pass
- Follow Python [Google code style](https://google.github.io/styleguide/pyguide.html)

## Code quality
- File and function docstrings should be written in [Google style](https://google.github.io/styleguide/pyguide.html)
- We use `black` to automatically format code in a standardized format. To ensure that any code changes are up to standard, use `pre-commit` as such.
```
# Run the following two lines ONCE.
pip install pre-commit
pre-commit install
```
Then, all future commits will call `black` automatically to format the code. Any code that does not follow the standard will cause a check to fail.

## Unit testing
Unit-tests should be written for most functions. To run unit tests, simply run the following.
```
# Install ONCE.
pip install -r dev-requirements.txt

# Run test
make test
```
Any failing tests will cause a check to fail.

## Documentation
We use `sphinx` to generate documentation. 
Importantly, we used the submodule functionality to import documentation from a separate repository (https://github.com/aristoteleo/spateo-tutorials).
It is important to keep the submodule up to date with the main repository and the following commands will help you do so.


### Update All Submodules at Once:

1. **Fetch and Merge Changes for All Submodules**:
   
   You can fetch the latest changes for all submodules and merge them into your current checkouts of the submodules:

   ```bash
   git submodule update --remote --merge
   ```

2. **Commit the Updated Submodules**:

   This step is important because the parent repository tracks a specific commit of the submodule. By updating the submodule, the parent repository needs to be informed of the new commit to track.

   ```bash
   git add .
   git commit -m "Updated all submodules"
   git push
   ```
Once you finish the above, check the link directory (something like spateo-tutorials @ 8e372ee) under the `docs` folder to make sure the related commit (such as 8e372ee) is the same as the latest one in the spateo-tutorials repository. If not, you may need to redo the above procedure again. 

