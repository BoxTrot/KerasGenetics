# KerasGenetics


Use asexual reproduction with mutation

-----------------------------------

Prototype data structure to hold the genetic material for each agent:
```
genes = [
    {
        "layer_type": "Dense",
        "config": "config"
    },
    {
        "layer_type": "sequential_functional",
        "contents": [
            {
                "layer_type": "Dense",
                "config": "config"
            },
            {
                "layer_type": "Dropout",
                "config": "config"
            }
        ]
    }
]
```

-----------------------------------

File format of "all_5_node_graphs.pickle" and "all_6_node_graphs.pickle":
Contains a 3d numpy ndarray. Iterate along the first axis to get each valid graph.
Each graph has node 0 as the input node, and the last node as the output node.