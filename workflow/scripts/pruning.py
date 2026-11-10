import copy
import networkx as nx
import plotly.express as px
import numpy as np
import slam.io as sio
import slam.texture as stex
import slam.plot as splt
import plotly.graph_objects as go
from collections import Counter
import potpourri3d as pp3d


import pandas as pd
import numpy as np

# Métadonnées issues des wildcards
test = snakemake.wildcards.test
sub  = snakemake.wildcards.sub
side = snakemake.wildcards.side   # hémisphère


#def calcul matrice de distance entre paire de feuille
## params:
### in: endpoints
### out: mat

#def trier la matrice par paire de distance croissante
## params:
### in: mat
### out: mat triée

#def pruning
## par paire de distance croissante, vérifier si les feuilles sont connectées par une bifurcation
### si oui, et si la distance <= seuil distance autorisée supprimer le segment le plus court
## params:
### in: endpoints, bifurcations, segment, graph, mat triée
### out: pruned graph