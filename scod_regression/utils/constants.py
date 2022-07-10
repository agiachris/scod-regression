_BASE_SCOD_CONFIG = {
    "output_dist": "NormalMeanParamLayer",  # output distribution determining Fischer
    "use_empirical_fischer": False,  # weight sketch samples by loss
    "sketch": "SinglePassPCA",  # sketch class (Gaussian or SRFT)
    "num_eigs": 10,  # low rank estimate to recover (k)
    "num_samples": None,  # sketch size (T)
}
