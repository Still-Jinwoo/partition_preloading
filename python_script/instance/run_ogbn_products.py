from celeritas.utils.preprocessing.dataset.ogbn_products import OGBNProducts
import utils.executor as e
import utils.report_result as r
from pathlib import Path

def run_ogbn_products(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):

    dataset_name = "ogbn_products"

    products_config_path = Path("/home/jinwoo/Celeritas/python_script/instance/configs_yaml/ogbn_products/ogbn_products.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Dataset {} is not on local, downloading... =====".format(dataset_name))
        dataset = OGBNProducts(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess(num_partitions=64, sequential_train_nodes=True)
    else:
        print("==== {} already existed and preprocessed =====".format(dataset_name))
    
    for i in range(num_runs):
        e.run_config(products_config_path, results_dir / Path("ogbn_products/celeritas_products"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "celeritas")
    
    r.print_results_summary([results_dir / Path("ogbn_products/celeritas_products")])
