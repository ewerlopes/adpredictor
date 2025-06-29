from adpredictor import AdPredictor, AdPredictorConfig
from collections import namedtuple
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import protobufs.adpredictor_pb2 as pb
import util as util
import os
import argparse

logger = logging.getLogger(__name__)


class Sampler(object):
    """Class that manages sampling feature_vector/label pairs."""

    def __init__(
        self,
        num_features,
        feature_cardinality,
        biased_feature_proportion,
        prior_probability,
        biased_feature_effect_length,
    ):
        self._num_features = num_features
        self._feature_cardinality = feature_cardinality
        self._biased_feature_proportion = biased_feature_proportion
        self._prior_probability = prior_probability
        self._biased_feature_effect_length = biased_feature_effect_length
        self._biased_weights = self._construct_biased_weights()
        self._num_samples = 0

    def __iter__(self):
        """Implementing the Python iterator protocol"""
        return self

    def __next__(self):
        """Implementing the Python iterator protocol"""
        feature_vector = [
            pb.Feature(feature=f, value=np.random.randint(0, self._cardinality(f)))
            for f in range(self._num_features)
        ]
        label = self._label(feature_vector)
        self._num_samples += 1
        return (feature_vector, label)

    def get_bias_for_feature(self, feature):
        return self._biased_weights.get(util.serialize_feature(feature))

    def _construct_biased_weights(self):
        biased_weights = {}
        for feature, value in itertools.product(
            range(1, self._num_features),
            range(self._feature_cardinality),
        ):
            key = util.serialize_feature(pb.Feature(feature=feature, value=value))
            if np.random.rand() < self._biased_feature_proportion:
                direction = np.random.rand() < self._prior_probability
                biased_weights[key] = direction
                logger.info(
                    "Biased truth feature (%s, %s) to %s", feature, value, direction
                )
        return biased_weights

    def _cardinality(self, f):
        return 1 if f == 0 else self._feature_cardinality

    def _biased_weights_label(self, features):
        for f in np.random.permutation(features):
            feature_weight = self.get_bias_for_feature(f)
            if feature_weight is not None:
                logger.debug(
                    "Hit in biased_weights (%s, %s) with bias %s",
                    f.feature,
                    f.value,
                    feature_weight,
                )
                return feature_weight
        logger.debug("Missed in biased weights")
        return None

    def _default_label(self):
        return np.random.rand() < self._prior_probability

    def _label(self, features):
        if self._num_samples > self._biased_feature_effect_length:
            return self._default_label()
        biased_label = self._biased_weights_label(features)
        if biased_label is not None:
            return biased_label
        return self._default_label()


SimulationConfig = namedtuple(
    "SimulationConfig",
    [
        "predictor_config",
        "feature_cardinality",
        "num_examples",
        "biased_feature_proportion",
        "out_directory",
        "graph_out_extension",
        "visualization_interval",
        "biased_feature_effect_length",
    ],
)

COLORS = plt.get_cmap("tab10").colors


def get_current_weights_by_feature(predictor):
    def by_feature(kv):
        return kv[0].feature

    def by_feature_value(kv):
        return (kv[0].feature, kv[0].value)

    weights = sorted(predictor.weights, key=by_feature_value)
    for feature, group in itertools.groupby(weights, key=by_feature):
        yield feature, [(f, w.mean, w.variance) for (f, w) in group]


def plot(predictor, sampler, num_examples, graph_out_extension, output_directory):
    plt.clf()

    # plot the current weights
    for color, (feature, weights) in zip(
        itertools.cycle(COLORS), get_current_weights_by_feature(predictor)
    ):
        _, means, variances = zip(*weights)
        logging.debug("Feature %s, Weights: %s", feature, weights)

        label = "F{}".format(feature) if feature != 0 else "Bias"
        plt.scatter(means, variances, label=label, color=color, alpha=0.8, s=40)

    # annotate biased weights
    for _, weights in get_current_weights_by_feature(predictor):
        for feature, mean, variance in weights:
            bias_weight = sampler.get_bias_for_feature(feature)
            if bias_weight is not None:
                plt.annotate("+" if bias_weight else "-", (mean, variance), size=40)

    plt.title("(μ, σ²) after {} examples".format(num_examples))
    plt.xlabel("μ")
    plt.ylabel("σ²")
    plt.legend(loc="best")
    plt.xlim(-4, 4)
    plt.ylim(-0.1, 1.1)

    filename = "{:03d}.{}".format(num_examples, graph_out_extension)
    logger.info("Saving graph to %s", filename)
    plt.savefig(os.path.join(output_directory, filename), dpi=300)


def main(
    verbose=False,
    beta=0.05,
    prior_probability=0.5,
    epsilon=0.05,
    num_features=8,
    feature_cardinality=5,
    num_examples=100,
    visualization_interval=100,
    biased_feature_proportion=0.2,
    biased_feature_effect_length=10**100,
    out_directory="./logs/adpredictor/",
    graph_out_extension="png",
):
    """Set up and run the AdPredictor on a simulated dataset.

    Args:
        verbose (bool): Enable verbose logging.
        beta (float): Beta parameter for the AdPredictor.
        prior_probability (float): Prior probability for the AdPredictor.
        epsilon (float): Epsilon parameter for the AdPredictor.
        num_features (int): Number of features in the dataset.
        feature_cardinality (int): Cardinality of each feature.
        num_examples (int): Number of examples to simulate.
        visualization_interval (int): Interval for visualizing the training progress.
        biased_feature_proportion (float): Proportion of features that are biased.
        biased_feature_effect_length (int): Length of the biased feature effect.
        out_dir (str): Directory to save output graphs.
        graph_out_extension (str): File extension for output graphs.

    Returns:
        None

    """
    # Initialize globals
    np.random.seed(1)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    # Construct settings
    predictor_config = AdPredictorConfig(
        beta=beta,
        prior_probability=prior_probability,
        epsilon=epsilon,
        num_features=num_features,
    )

    simulation_config = SimulationConfig(
        predictor_config=predictor_config,
        feature_cardinality=feature_cardinality,
        num_examples=num_examples,
        out_directory=out_directory,
        biased_feature_proportion=biased_feature_proportion,
        biased_feature_effect_length=biased_feature_effect_length,
        visualization_interval=visualization_interval,
        graph_out_extension=graph_out_extension,
    )

    predictor = AdPredictor(simulation_config.predictor_config)
    sampler = Sampler(
        num_features=simulation_config.predictor_config.num_features,
        feature_cardinality=simulation_config.feature_cardinality,
        biased_feature_proportion=simulation_config.biased_feature_proportion,
        prior_probability=simulation_config.predictor_config.prior_probability,
        biased_feature_effect_length=biased_feature_effect_length,
    )

    # Train and output graphs
    samples = itertools.islice(sampler, simulation_config.num_examples)
    for iteration, (features, label) in enumerate(samples):
        predictor.train(features, label)
        if iteration % simulation_config.visualization_interval == 0:
            plot(
                predictor=predictor,
                sampler=sampler,
                num_examples=iteration,
                graph_out_extension=simulation_config.graph_out_extension,
                output_directory=simulation_config.out_directory,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AdPredictor simulation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--beta", type=float, default=0.05, help="Beta parameter")
    parser.add_argument(
        "--prior-probability", type=float, default=0.5, help="Prior probability"
    )
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon parameter")
    parser.add_argument(
        "--num-features", type=int, default=8, help="Number of features"
    )
    parser.add_argument(
        "--feature-cardinality", type=int, default=5, help="Feature cardinality"
    )
    parser.add_argument(
        "--num-examples", type=int, default=100, help="Number of examples"
    )
    parser.add_argument(
        "--visualization-interval", type=int, default=100, help="Visualization interval"
    )
    parser.add_argument(
        "--biased-feature-proportion",
        type=float,
        default=0.2,
        help="Biased feature proportion",
    )
    parser.add_argument(
        "--biased-feature-effect-length",
        type=int,
        default=10**100,
        help="Biased feature effect length",
    )
    parser.add_argument(
        "--out-directory",
        type=str,
        default="./logs/adpredictor/",
        help="Output directory",
    )
    parser.add_argument(
        "--graph-out-extension",
        type=str,
        default="png",
        help="Output extension for graphs",
    )

    args = parser.parse_args()

    main(
        verbose=args.verbose,
        beta=args.beta,
        prior_probability=args.prior_probability,
        epsilon=args.epsilon,
        num_features=args.num_features,
        feature_cardinality=args.feature_cardinality,
        num_examples=args.num_examples,
        visualization_interval=args.visualization_interval,
        biased_feature_proportion=args.biased_feature_proportion,
        biased_feature_effect_length=args.biased_feature_effect_length,
        out_directory=args.out_directory,
        graph_out_extension=args.graph_out_extension,
    )
