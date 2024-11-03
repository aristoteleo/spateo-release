<p align="center">
  <img height="150" src="https://raw.githubusercontent.com/aristoteleo/spateo-release/main/docs/_static/logo.png" />
</p

[![documentation](https://readthedocs.org/projects/spateo-release/badge/?version=latest)](https://spateo-release.readthedocs.io/en/latest/)


[Installation](https://spateo-release.readthedocs.io/en/latest/installation.html) - [Tutorials](https://spateo-release.readthedocs.io/en/latest/tutorials/index.html) - [API](https://spateo-release.readthedocs.io/en/latest/autoapi/spateo/index.html) - [Citation](https://www.biorxiv.org/content/10.1101/2022.12.07.519417v1) - [Technical](https://spateo-release.readthedocs.io/en/latest/technicals/index.html)

# Citation
Xiaojie Qiu1, 7, 8$*, Daniel Y. Zhu3$, Yifan Lu1, 7, 8, 9$, Jiajun Yao2, 4, 10$, Zehua Jing2, 4, 11$, Kyung Hoi (Joseph) Min12$, Mengnan Cheng2，6$, Hailin Pan6, Lulu Zuo6, Samuel King13, Qi Fang2, 6, Huiwen Zheng2, 11, Mingyue Wang2, 14, Shuai Wang2, 11, Qingquan Zhang25, Sichao Yu5, Sha Liao6, 17, 18, Chao Liu15, Xinchao Wu2, 4, 16, Yiwei Lai6, Shijie Hao2, Zhewei Zhang2, 4, 16, Liang Wu18, Yong Zhang15, Mei Li17, Zhencheng Tu2, 11, Jinpei Lin2, 4, Zhuoxuan Yang2, 16, Yuxiang Li15, Ying Gu2, 6, 11, Ao Chen6, 17, 18, Longqi Liu2, 19, 20, Jonathan S. Weissman5, 22, 23, Jiayi Ma9*, Xun Xu2, 11, 21*, Shiping Liu2, 19, 20, 24*, Yinqi Bai4, 26*  $Co-first authors; *:Corresponding authors
 
Spatiotemporal modeling of molecular holograms 

https://www.biorxiv.org/content/10.1101/2022.12.07.519417v1

# Abstract

<p align="justify">
Quantifying spatiotemporal dynamics during embryogenesis is crucial for understanding congenital diseases. We developed Spateo, a 3D spatiotemporal modeling framework, and applied it to a 3D mouse embryogenesis atlas at E9.5 and E11.5, capturing eight million cells. Spateo enables scale, partial, non-rigid alignment, multi-slice refinement and mesh correction to create molecular holograms of whole embryos. It introduces digitization methods to uncover multi-level biology from subcellular to whole-organ, identifying expression gradients along orthogonal axes of emergent 3D structures, e.g. secondary organizers such as MHB. Spateo further jointly models intercellular and intracellular interaction to dissect signaling landscapes in 3D structures, including the ZLI. Lastly, Spateo introduces “morphometric vector fields” of cell migration, integrates spatial differential geometry to unveil molecular programs underlying asymmetrical murine heart organogenesis and others, bridging macroscopic changes with molecular dynamics. Thus, Spateo enables the study of organ ecology at a molecular level in 3D space over time.
</p>

# Keywords

<p align="justify">
Spateo, whole embryo 3D spatial transcriptomics, 3D reconstruction, Stereo-seq, spatial domain digitization, ligand receptor cell-cell interactions, intercellular and intracellular interactions, organogenesis mode, morphometric vector field, spatial differential geometry analyses
</p>

![Spateo](https://github.com/user-attachments/assets/9581284c-0617-4561-8827-81134618dabf)

## Highlights of Spateo:

*  <p align="justify"> Spateo introduces a sophisticated approach, Starro, to segment single cells based purely on RNA signal, unsupervisedly identifies continuous tissue domains via spatially-constrained clustering, and dissect the intricate spatial cell type distribution and tissue composition;

* <p align="justify"> Spateo identifies spatial polarity/gradient genes (e.g. neuronal layer specific genes) by solving a partial differential equation to digitize layers and columns of a spatial domain. </p

* <p align="justify"> Spateo implements a full suite of spatially-aware modules for differential expression inference, including novel parametric models for spatially-informed prediction of cell-cell interactions and interpretable estimation of downstream effects. </p

* <p align="justify"> Spateo enables reconstruction of 3D whole-organ models from 2D slices, identifying different “organogenesis modes” (patterns of cell migration during organogenesis) for each organ and quantifying morphometric properties (such as organ surface area, volume, length and cell density) over time. </p

* <p align="justify"> Spateo brings in the concept of the “morphometric vector field” that predicts migration paths for each cell within an organ in a 3D fashion and reveals principles of cell migration by exploring various differential geometry quantities. </p

## Spateo Development Process
- Follow feature-staging-main review process
    - create a specific branch for new feature
    - implement and test on your branch; add unit tests
    - create pull request
    - discuss with lab members and merge into the main branch once all checks pass
- Follow python [Google code style](https://google.github.io/styleguide/pyguide.html)

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
Once you are done the above, check the link directory (something like spateo-tutorials @ 8e372ee) under the `docs` folder to make sure the related commit (such as 8e372ee) is the same as the latest one in the spateo-tutorials repository. If not, you may need to redo the above procedure again. 
```
