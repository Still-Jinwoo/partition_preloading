# import torch
# import torch.profiler
# from torch.profiler import profile, ProfilerActivity
from celeritas.utils.preprocessing.dataset.ogbn_arxiv import OGBNArxiv
import utils.executor as e
import utils.report_result as r
from pathlib import Path

def run_ogbn_arxiv(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
    #              profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/ogbn_arxiv')) as prof:
    dataset_name = "ogbn_arxiv"

    arxiv_config_path = Path("/home/jinwoo/Celeritas/python_script/instance/configs_yaml/ogbn_arxiv/ogbn_arxiv.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Dataset {} is not on local, downloading... =====".format(dataset_name))
        dataset = OGBNArxiv(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess(num_partitions=32, sequential_train_nodes=True)
    else:
        print("==== {} already existed and preprocessed =====".format(dataset_name))
    
    for i in range(num_runs):
        e.run_config(arxiv_config_path, results_dir / Path("ogbn_arxiv/celeritas_arxiv"),
                    overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "celeritas")


    r.print_results_summary([results_dir / Path("ogbn_arxiv/celeritas_arxiv")])
