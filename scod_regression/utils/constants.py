_BASE_SCOD_CONFIG = {
    # Output distribution
    "output_dist": "NormalMeanParamLayer",
    "use_empirical_fischer": False,   # weight sketch samples by loss
    # Matrix sketching
    "sketch": "SinglePassPCA",        # sketch class (SinglePass)
    "num_eigs": 10,                   # low rank estimate to recover (k)    
    "num_samples": None,              # sketch size (T)
}
