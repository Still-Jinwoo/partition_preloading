import argparse
import os
# import torch
# import torch.profiler
# from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from instance.run_obgn_arxiv import run_ogbn_arxiv
from instance.run_ogbn_paper100M import run_ogbn_paper100M
from instance.run_ogbn_products import run_ogbn_products

# 기본 데이터셋 및 결과 디렉토리 경로 설정
DEFAULT_DATASET_DIRECTORY = "datasets/"
DEFAULT_RESULTS_DIRECTORY = "results/"

if __name__ == "__main__":
    # 실행할 실험과 그에 해당하는 함수를 딕셔너리로 매핑
    experiment_dict = {
        "instance_arxiv": run_ogbn_arxiv,
        "instance_papers100m": run_ogbn_paper100M,
        "instance_products": run_ogbn_products,
    }
    # 커맨드 라인 인자 파서를 설정하여 사용자로부터 입력을 받음
    parser = argparse.ArgumentParser(description='Reproduce experiments ')
    parser.add_argument('--experiment', metavar='experiment', type=str, choices=experiment_dict.keys(),
                        help='Experiment choices: %(choices)s')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='If true, the results of previously run experiments will be overwritten.')
    parser.add_argument('--enable_dstat', dest='enable_dstat', action='store_true',
                        help='If true, dstat resource utilization metrics.')
    parser.add_argument('--enable_nvidia_smi', dest='enable_nvidia_smi', action='store_true',
                        help='If true, nvidia-smi will collect gpu utilization metrics.')
    parser.add_argument('--dataset_dir', metavar='dataset_dir', type=str, default=DEFAULT_DATASET_DIRECTORY,
                        help='Directory containing preprocessed dataset(s). If a given dataset is not present'
                            ' then it will be downloaded and preprocessed in this directory')
    parser.add_argument('--results_dir', metavar='results_dir', type=str, default=DEFAULT_RESULTS_DIRECTORY,
                        help='Directory for output of results')
    parser.add_argument('--show_output', dest='show_output', action='store_true',
                        help='If true, the output of each run will be printed directly to the terminal.')
    parser.add_argument('--short', dest='short', action='store_true',
                        help='If true, a shortened version of the experiment(s) will be run')
    parser.add_argument('--num_runs', dest='num_runs', type=int, default=1,
                        help='Number of runs for each configuration. Used to average results.')
    # 인자를 파싱하여 args 객체에 저장
    args = parser.parse_args()

     # 경로를 Path 객체로 변환하여 플랫폼에 독립적인 경로 관리를 용이하게 함
    args.dataset_dir = Path(args.dataset_dir)
    args.results_dir = Path(args.results_dir)

    # 데이터셋 및 결과 디렉토리가 존재하지 않을 경우 생성
    if not args.dataset_dir.exists():
        os.makedirs(args.dataset_dir)
    if not args.results_dir.exists():
        os.makedirs(args.results_dir)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
    #              profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18')) as prof:

    # 각 실험을 선택하여 해당 함수 실행
    experiment_dict.get(args.experiment)(args.dataset_dir,
                                        args.results_dir,
                                        args.overwrite,
                                        args.enable_dstat,
                                        args.enable_nvidia_smi,
                                        args.show_output,
                                        args.short,
                                        args.num_runs)
