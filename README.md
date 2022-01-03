# spateo-release
Spatiotemporal modeling of spatial transcriptomics 


## Spateo Development Process
- Follow feature-staging-main review process
    - create a specific branch for new feature
    - implement and test on your branch
    - create pull request
    - discuss with lab members and merge into the main branch
- Follow python [google code style](https://google.github.io/styleguide/pyguide.html)


## Install Precommit Hook  
**You would like to ensure that the code changes you push have good code style**  
**This step enables pre-commit to check and auto-format your style before every git commit locally via commandline.**
### Install (once)  
`pip install pre-commit`  
`pre-commit install`  
### format all code (rare usage)  
`pre-commit run --all`  
Now everytime you run `git commit -m msg`, pre-commit will check your changes.
