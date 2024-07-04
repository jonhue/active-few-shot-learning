import torch
from afsl.acquisition_functions import AcquisitionFunction
from afsl.acquisition_functions.cosine_similarity import CosineSimilarity
from afsl.acquisition_functions.ctl import CTL
from afsl.acquisition_functions.information_density import InformationDensity
from afsl.acquisition_functions.itl import ITL
from afsl.acquisition_functions.kmeans_pp import KMeansPP
from afsl.acquisition_functions.least_confidence import LeastConfidence
from afsl.acquisition_functions.max_dist import MaxDist
from afsl.acquisition_functions.max_entropy import MaxEntropy
from afsl.acquisition_functions.min_margin import MinMargin
from afsl.acquisition_functions.random import Random
from afsl.acquisition_functions.uncertainty_sampling import UncertaintySampling
from afsl.acquisition_functions.undirected_itl import UndirectedITL
from afsl.acquisition_functions.undirected_vtl import UndirectedVTL
from afsl.acquisition_functions.vtl import VTL
from afsl.utils import DEFAULT_MINI_BATCH_SIZE


def get_acquisition_function(
    alg: str,
    target: torch.Tensor,
    noise_std: float,
    num_workers: int,
    subsample_acquisition: bool,
    mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE,
    subsampled_target_frac: float = 1.0,
    max_target_size: int | None = None,
    force_nonsequential: bool = False,
) -> AcquisitionFunction:
    if alg == "Random" or alg == "OracleRandom":
        acquisition_function = Random(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
        )
    elif alg == "ITL" or alg == "ITL-nonsequential":
        acquisition_function = ITL(
            target=target,
            noise_std=noise_std,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=force_nonsequential or (alg == "ITL-nonsequential"),
        )
    elif alg == "VTL":
        acquisition_function = VTL(
            target=target,
            noise_std=noise_std,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=force_nonsequential,
        )
    elif alg == "CTL":
        acquisition_function = CTL(
            target=target,
            noise_std=noise_std,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=force_nonsequential,
        )
    elif alg == "CosineSimilarity":
        acquisition_function = CosineSimilarity(
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
        )
    elif alg == "InformationDensity":
        acquisition_function = InformationDensity(
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
        )
    elif alg == "UndirectedITL":
        acquisition_function = UndirectedITL(
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=force_nonsequential,
        )
    elif alg == "UndirectedVTL":
        acquisition_function = UndirectedVTL(
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=force_nonsequential,
        )
    elif alg == "UncertaintySampling":
        acquisition_function = UncertaintySampling(
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
        )
    elif alg == "MinMargin":
        acquisition_function = MinMargin(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
        )
    elif alg == "MaxEntropy":
        acquisition_function = MaxEntropy(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
        )
    elif alg == "LeastConfidence":
        acquisition_function = LeastConfidence(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
        )
    elif alg == "MaxDist":
        acquisition_function = MaxDist(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=force_nonsequential,
        )
    elif alg == "KMeansPP":
        acquisition_function = KMeansPP(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=force_nonsequential,
        )
    else:
        raise NotImplementedError

    return acquisition_function
