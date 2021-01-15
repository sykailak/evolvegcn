EvolveGCN
=====

This repository contains the code that was mildly modified from [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191), published in AAAI 2020.




#### install StellarGraph

StellarGraph will be used for creating initial embedding of nodes. 
If running on colab, run the following code.

```python
import sys
if 'google.colab' in sys.modules:
  %pip install -q stellargraph[demos]==1.2.1

# verify that we're using the correct version of StellarGraph for this notebook
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None
```

#### Generate edge data from the betting data

Run the following on the console.

```python
python create_edges.py --data_path '/content/drive/My Drive/KindredEvolve/rec_bet_events2.csv' --save_path '/content/drive/My Drive/KindredEvolve/tennis_edge.csv' --sport 'tennis'
```

These are the arguments:
-	data_path: the directory where betting data is stored (In our case, the betting data is ‘rec_bet_events’)
-	save_path: the directory where the generated edge data will be stored
-	sport: the sport in question

The generated edge data has information of edges, with corresponding timestamps. It will have following features:

-	source: the index of the source node (users)
-	target: the index of the target node (major-leagues)
-	weight: ‘1’ denotes there exists an edge. 
-	time: the corresponding timestamp where the edge exists

#### Run EvolveGCN
I have came up with the best hyperparameters for each sport. Run the following in the console:

For football
```python
python run_exp.py --edge_file '/content/drive/My Drive/KindredEvolve/football_edge.csv' --sport football --comment 'you can insert comment here' --adaptive --adj_mat_time_window 1 --num_hist_steps 0
```

For ice hockey
```python
python run_exp.py --edge_file '/content/drive/My Drive/KindredEvolve/icehockey_edge.csv' --sport icehockey --comment 'you can insert comment here' --noadaptive --adj_mat_time_window 1 --num_hist_steps 0
```

For basketball
```python
python run_exp.py --edge_file '/content/drive/My Drive/KindredEvolve/basketball_edge.csv' --sport basketball --comment 'you can insert comment here' --noadaptive --adj_mat_time_window 1 --num_hist_steps 0
```
	
For tennis
```python
python run_exp.py --edge_file '/content/drive/My Drive/KindredEvolve/tennis_edge.csv' --sport tennis --comment 'you can insert comment here' --noadaptive --adj_mat_time_window 1 --num_hist_steps 0
```

These are the arguments:
-	edge_file: directory where the edge data is stored (it is created in #3)
-	sport: the sport in question
-	comment: comment that appears at the top
-	adaptive/noadaptive: stores true/false according to whether one wants to use exact class weight 
-	adj_mat_time_window: time window to create the adjacent matrix for each timestep
-	num_hist_steps: number of historical steps to use in RNN


EvolveGCN gives the following mAPs (mean Average Precision):
-	football: 0.7723
-	ice hockey: 0.7967
-	basketball: 0.8155
-	tennis: 0.7989

