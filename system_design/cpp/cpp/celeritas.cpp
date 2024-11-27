#include "celeritas.h"

#include "config.h"
#include "evaluator.h"
#include "io_manager.h"
#include "logger.h"
#include "model.h"
#include "train.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>

// celeritas 함수: 프로그램의 주요 흐름을 관리하는 함수
void celeritas(int argc, char *argv[]) {
    // Celeritas 실행 시작 시간 측정
    auto celeritas_start = std::chrono::high_resolution_clock::now(); 

    bool train = true; // 학습 모드인지 평가 모드인지 결정
    string command_path = string(argv[0]);
    string config_path = string(argv[1]); // 설정 파일 경로를 가져옴

    string command_name = command_path.substr(command_path.find_last_of("/\\") + 1);

    // 실행 파일 이름이 celeritas_eval이면 평가 모드로 전환
    if (strcmp(command_name.c_str(), "celeritas_eval") == 0) { 
        train = false;
    }

    // 설정 파일을 읽어 초기화 (데이터 흐름의 시작)
    shared_ptr<CeleritasConfig> celeritas_config = initConfig(config_path);

    // Random Seed실행 시작 시간 측정
    auto seed_start = std::chrono::high_resolution_clock::now(); 
    
    // 재현 가능성을 위해 랜덤 시드 설정
    torch::manual_seed(celeritas_config->model->random_seed);
    srand(celeritas_config->model->random_seed);

    // Random Seed 종료 시간 측정
    auto seed_end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> seed_seconds = seed_end - seed_start; 
    std::cout << "Random Seed Time Taken: " << seed_seconds.count() << "s\n"; 

    // 초기화 타이머 시작
    Timer initialization_timer = Timer(false);
    initialization_timer.start();
    SPDLOG_INFO("Start initialization");

    // 사용 장치(CPU/GPU) 설정
    std::vector<torch::Device> devices;
    if (celeritas_config->storage->device_type == torch::kCUDA) { // GPU 사용 여부 확인
        std::cout << "@@@ Celeritas is using GPU @@@ \n";
        for (int i = 0; i < celeritas_config->storage->device_ids.size(); i++) {
            devices.emplace_back(torch::Device(torch::kCUDA, celeritas_config->storage->device_ids[i]));
        }
        if (devices.empty()) { // GPU가 없다면 기본 GPU(0번 장치) 사용
            devices.emplace_back(torch::Device(torch::kCUDA, 0));
        }
    } else {
        std::cout << "@@@ Celeritas is using CPU @@@ \n";
        devices.emplace_back(torch::kCPU); // GPU가 없으면 CPU 사용
    }
    // 모델 초기화 (데이터 흐름에서 모델을 사용하기 위한 준비)
    std::shared_ptr<Model> model = initializeModel(celeritas_config->model,
                                                   devices,
                                                   celeritas_config->storage->dataset->num_relations);
    model->train_ = train; // 학습 모드 설정


    // 평가 설정에서 Negative Sampling이 있는 경우 필터링 설정
    if (celeritas_config->evaluation->negative_sampling != nullptr) {
        std::cout << "@@ Negative Sampling: True \n";
        model->filtered_eval_ = celeritas_config->evaluation->negative_sampling->filtered;
    } else {
        std::cout << "@@ Negative Sampling: False \n";
        model->filtered_eval_ = false;
    }
    // Graph Model Storage Initialization 실행 시작 시간 측정    
    auto graph_init_start = std::chrono::high_resolution_clock::now(); 

    // 스토리지 초기화: 데이터 로딩 및 I/O 작업의 주요 부분
    GraphModelStorage *graph_model_storage = initializeStorage(model, celeritas_config->storage); // 스토리지 이니셜라이즈

    // Train 종료 시간 측정
    auto graph_init_end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> graph_init_seconds = graph_init_end - graph_init_start; 
    std::cout << "Graph Model Storage Initialization Time: " << graph_init_seconds.count() << "s\n"; 

    // Graph Model Storage Initialization 실행 시작 시간 측정    
    auto dataloader_start = std::chrono::high_resolution_clock::now(); 

    // DataLoader 초기화: 학습에 필요한 데이터 배치를 관리
    DataLoader *dataloader = new DataLoader(graph_model_storage,
                                            celeritas_config->training,
                                            celeritas_config->evaluation,
                                            celeritas_config->model->encoder);
    
    // dataloader 종료 시간 측정
    auto dataloader_end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> dataloader_seconds = dataloader_end - dataloader_start; 
    std::cout << "Dataloader time taken: " << dataloader_seconds.count() << "s\n"; 

    // 초기화 타이머 종료 및 시간 로깅
    initialization_timer.stop();
    int64_t initialization_time = initialization_timer.getDuration();
    SPDLOG_INFO("Initialization Complete: {}s", (double) initialization_time / 10000);

    // 학습 및 평가 객체 생성
    Trainer *trainer;
    Evaluator *evaluator;

    // Train 실행 시작 시간 측정
    auto train_start = std::chrono::high_resolution_clock::now(); 
    // 학습 모드일 때
    if (train) {
        // 동기식 또는 파이프라인 방식의 학습 방식 결정
        if (celeritas_config->training->pipeline->sync) {
            if (celeritas_config->storage->device_ids.size() > 1) {
                trainer = new SynchronousMultiGPUTrainer(dataloader, model, celeritas_config->training->logs_per_epoch);
            } else {
                trainer = new SynchronousTrainer(dataloader, model, celeritas_config->training->logs_per_epoch);
            }
        } else {
            trainer = new PipelineTrainer(dataloader,
                                          model,
                                          celeritas_config->training->pipeline,
                                          celeritas_config->training->logs_per_epoch);
        }

        // 동기식 또는 파이프라인 방식의 평가 방식 결정
        if (celeritas_config->evaluation->pipeline->sync) {
            evaluator = new SynchronousEvaluator(dataloader, model);
        } else {
            evaluator = new PipelineEvaluator(dataloader,
                                              model,
                                              celeritas_config->evaluation->pipeline);
        }

        // 학습 및 평가 루프
        for (int epoch = 0; epoch < celeritas_config->training->num_epochs; epoch++) {
            if ((epoch + 1) % celeritas_config->evaluation->epochs_per_eval != 0) {
                trainer->train(1);
            } else {
                trainer->train(1);
                evaluator->evaluate(true); // 평가 (검증 데이터셋)
                evaluator->evaluate(false); // 평가 (테스트 데이터셋)
            }
        }
    } else {
        // 평가 모드일 때
        if (celeritas_config->evaluation->pipeline->sync) {
            evaluator = new SynchronousEvaluator(dataloader, model);
        } else {
            evaluator = new PipelineEvaluator(dataloader,
                                              model,
                                              celeritas_config->evaluation->pipeline);
        }
        evaluator->evaluate(false); // 평가 (테스트 데이터셋)
    }
    // Train 종료 시간 측정
    auto train_end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> train_seconds = train_end - train_start; 
    std::cout << "Total Train Time: " << train_seconds.count() << "s\n"; 

    // model save 실행 시작 시간 측정
    auto model_save_start = std::chrono::high_resolution_clock::now(); 

    // 학습된 모델 저장
    model->save(celeritas_config->storage->dataset->base_directory); // model save?

    // Model Save 종료 시간 측정
    auto  model_save_end = std::chrono::high_resolution_clock::now();  
    std::chrono::duration<double> model_save_seconds = model_save_end - model_save_start;  
    std::cout << "Total Model Save Time: " << model_save_seconds.count() << "s\n"; 

    // GC 실행 시작 시간 측정
    auto GC_start = std::chrono::high_resolution_clock::now(); 

    // 메모리 정리
    delete graph_model_storage;
    delete trainer;
    delete evaluator;
    delete dataloader;

    // GC 종료 시간 측정
    auto GC_end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> GC_seconds = GC_end - GC_start; 
    std::cout << "Total GC Time: " << GC_seconds.count() << "s\n"; 

    // Celeritas 종료 시간 측정
    auto celeritas_end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = celeritas_end - celeritas_start; 
    std::cout << "Celeritas Total Time: " << elapsed_seconds.count() << "s\n";
}

// 메인 함수: celeritas 함수 호출로 실행 시작
int main(int argc, char *argv[]) {
    celeritas(argc, argv);
}