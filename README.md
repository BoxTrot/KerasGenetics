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
