# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""

Main run script for OpenFold3. Please see the README for usage details.

"""
# ruff: noqa: F821

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sys, os
if sys.platform != 'linux' and 'TORCH_LOGS' not in os.environ:
    # Suppress torch logging of "NOTE: Redirects are currently not supported in Windows or MacOs."
    os.environ['TORCH_LOGS'] = '-distributed'

from pathlib import Path

import click

from openfold3.core.config import config_utils
from openfold3.entry_points.import_utils import _torch_gpu_setup

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Yaml that specifies model and dataset parameters,"
    " see examples/training_new.yml",
)
@click.option("--seed", type=int, help="Initial seed for all processes")
@click.option(
    "--data-seed",
    "--data_seed",
    type=int,
    help="Initial seed for data pipeline. Defaults to seed if not specified.",
)
def train(runner_yaml: Path, seed: int | None = None, data_seed: int | None = None):
    """Perform a training experiment with a preprepared dataset cache."""
    _torch_gpu_setup()
    from openfold3.entry_points.experiment_runner import (
        TrainingExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        TrainingExperimentConfig,
    )

    runner_dict = config_utils.load_yaml(runner_yaml)

    # overwrite seed defaults if provided:
    if seed is not None:
        runner_dict["experiment_settings"]["seed"] = seed

    if data_seed is not None:
        runner_dict["data_module_args"]["data_seed"] = data_seed

    expt_config = TrainingExperimentConfig.model_validate(runner_dict)

    expt_runner = TrainingExperimentRunner(expt_config)
    expt_runner.setup()
    expt_runner.run()

def validate_int_list(ctx, param, value):
    if value is None:
        return []
    try:
        return [int(x.strip()) for x in value.split(',')]
    except ValueError:
        raise click.BadParameter('Must be a comma-separated list of integers.')

@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--inference-ckpt-path",
    "--inference_ckpt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Path for model checkpoint to be used for inference. "
    "If not specified, will attempt to find or download parameters to "
    "$OPENFOLD_CACHE [default: ~/.openfold3/]",
)
@click.option(
    "--inference-ckpt-name",
    "--inference_ckpt_name",
    type=str,
    required=False,
    help="Name of the checkpoint to be used for inference."
    " Only used if `inference_ckpt_path` is not specified.",
)
@click.option(
    "--num-diffusion-samples",
    "--num_diffusion_samples",
    type=int,
    default=None,
    required=False,
    help="Number of diffusion samples to generate for each query.",
)
@click.option(
    "--num-model-seeds",
    "--num_model_seeds",
    type=int,
    default=None,
    required=False,
    help="Number of model seeds to use for each query.",
)
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option(
    "--use-msa-server",
    "--use_msa_server",
    type=bool,
    default=True,
    help="Use ColabFold MSA server to perform alignments.",
)
@click.option(
    "--use-templates",
    "--use_templates",
    type=bool,
    default=True,
    help="Use ColabFold MSA server to perform template alignments.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Output directory for writing results",
)
@click.option(
    "--msa_and_templates_only",
    type=bool,
    default=False,
    help="Use ColabFold MSA server to get MSAs and templates but don't predict structures.",
)
@click.option(
    "--device",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The device to use for prediction: gpu, cpu, tpu. Default is gpu.",
    default="gpu",
)
@click.option(
    "--precision",
    type=click.Choice(["bf16-mixed", "32-true"]),
    help="The floating point precision used by pytorch_lightning.Trainer. Default is bf16-mixed when using CUDA on Linux or Windows, otherwise 32-true.",
    default=None,
)
@click.option(
    "--seeds",
    type=str,
    callback=validate_int_list,
    help="Random number seeds for inference",
)
def predict(
    query_json: Path,
    inference_ckpt_path: Path | None = None,
    inference_ckpt_name: str | None = None,
    num_diffusion_samples: int | None = None,
    num_model_seeds: int | None = None,
    runner_yaml: Path | None = None,
    use_msa_server: bool = True,
    use_templates: bool = True,
    output_dir: Path | None = None,
    msa_and_templates_only: bool = False,
    device: str = "gpu",
    precision: str | None = None,
    seeds: list[int] | None = None,
):
    """Perform inference on a set of queries defined in the query_json."""
    _torch_gpu_setup()

    from openfold3.entry_points.experiment_runner import (
        InferenceExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        InferenceExperimentConfig,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    logging.basicConfig(level=logging.INFO)
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()

    expt_config = InferenceExperimentConfig(
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
        **runner_args,
    )
    expt_config.pl_trainer_args.accelerator = device

    if precision is None:
        precision = 32    # bfloat16 is many times slower than float32 except with Nvidia CUDA.
        if sys.platform in ('linux', 'win32') and device == 'gpu':
            import torch
            if torch.cuda.is_available():
                precision = "bf16-mixed"
    expt_config.pl_trainer_args.precision = precision

    if seeds:
        expt_config.experiment_settings.seeds = seeds

    expt_runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples,
        num_model_seeds,
        use_msa_server,
        use_templates,
        output_dir,
    )
    if device == 'cpu':
        expt_runner.model_config['settings']['memory']['eval']['use_deepspeed_evo_attention'] = False
    
    # Load inference query set
    query_set = InferenceQuerySet.from_json(query_json)

    if msa_and_templates_only:
        expt_runner.inference_query_set = query_set
        expt_runner.lightning_data_module.prepare_data()
        return
        
    # Run the forward pass
    expt_runner.setup()
    expt_runner.run(query_set)
    expt_runner.cleanup()


@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory for writing alignments",
)
@click.option(
    "--msa-computation-settings-yaml",
    "--msa_computation_settings_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml file to customize Colabfold MSA settings,"
    " see MsaComputationSettings for options.",
)
def align_msa_server(
    query_json: Path,
    output_dir: Path,
    msa_computation_settings_yaml: Path | None = None,
):
    """Run MSA server alignment only with ColabFold MSA server.
    
    Example command:
    python run_openfold.py align-msa-server \
        --query_json query_example.json \
        --output_dir output/msa_server_test \
    
    More settings can be specified using the `msa_computation_settings_yaml` flag
    An example yaml file is provided in `examples/msa_server.yml`
    """
    _torch_gpu_setup()
    from openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        preprocess_colabfold_msas,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    query_set = InferenceQuerySet.from_json(query_json)

    msa_settings = MsaComputationSettings.from_config_with_cli_override(
        output_dir, msa_computation_settings_yaml
    )
    query_set = preprocess_colabfold_msas(
        inference_query_set=query_set,
        compute_settings=msa_settings,
    )

    with open(output_dir / "query_msa.json", "w") as fp:
        fp.write(query_set.model_dump_json(indent=4))


if __name__ == "__main__":
    cli()
