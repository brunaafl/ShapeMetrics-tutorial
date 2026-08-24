"""Simulations behind Figure 1.

Three parameterisations of the same question -- what is each analysis sensitive to:

    region_space   neurons vary in tuning SHAPE (skew, width); the clustered
                   variables enter the tuning linearly, so population geometry
                   depends on the cloud's first two moments alone
    theta_space    the same design with the clustered variables made CONDITIONS
                   (preferred angles of a 2-D von Mises), which enter non-linearly
    grid_space     the 2-D design where within-region and across-region clustering
                   vary independently
"""
from . import grid_space, region_space, theta_space   # noqa: F401
