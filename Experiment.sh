#!/bin/bash 

python main.py --n-epochs=200 --dataset=cora --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False

python main.py --n-epochs=200 --dataset=citeseer --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False

python main.py --n-epochs=200 --dataset=pubmed --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=False --SR=False

python main.py --n-epochs=200 --dataset=cora --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=cora --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False

python main.py --n-epochs=200 --dataset=citeseer --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=citeseer --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False

python main.py --n-epochs=200 --dataset=pubmed --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False
python main.py --n-epochs=200 --dataset=pubmed --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=False

python main.py --n-epochs=200 --dataset=cora --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=cora --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=cora --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=cora --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=cora --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True

python main.py --n-epochs=200 --dataset=citeseer --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=citeseer --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=citeseer --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=citeseer --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=citeseer --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True

python main.py --n-epochs=200 --dataset=pubmed --gnn=gat --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=pubmed --gnn=gcn --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=pubmed --gnn=sgc --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=pubmed --gnn=graphsage --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True
python main.py --n-epochs=200 --dataset=pubmed --gnn=ppnp --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True