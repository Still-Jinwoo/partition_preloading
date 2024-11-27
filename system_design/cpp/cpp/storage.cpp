#include "storage.h"

#include <fcntl.h>
#include <unistd.h>

#include <iostream>

#include "constants.h"
#include "logger.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using std::ios;
using std::ios_base;
// 파일 이름 변경 함수
void renameFile(string old_filename, string new_filename) {
    int result = rename(old_filename.c_str(), new_filename.c_str());
    if (result != 0) {
        SPDLOG_ERROR("Unable to rename {}\nError: {}", old_filename, errno);
        throw std::runtime_error("");// 파일 이름 변경 실패 시 예외 발생
    }
}
// 파일 복사 함수
void copyFile(string src_file, string dst_file) {
    std::ifstream src;
    std::ofstream dst;

    src.open(src_file, ios::in | ios::binary);// 원본 파일 열기
    dst.open(dst_file, ios::out | ios::binary);// 복사할 파일 열기

    dst << src.rdbuf();// 파일 내용을 복사

    src.close();
    dst.close();
}
// PartitionBufferStorage 클래스 생성자: 데이터 저장과 관련된 초기 설정을 수행
PartitionBufferStorage::PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double) dim0_size_ / options_->num_partitions);
    // 파티션 버퍼 초기화: 메모리와 스토리지 간의 데이터 이동 관리
    buffer_ = new PartitionBuffer(options_->buffer_capacity,
                                  options_->num_partitions,
                                  options_->fine_to_coarse_ratio,
                                  partition_size,
                                  dim1_size_,
                                  dim0_size_,
                                  dtype_,
                                  filename_,
                                  options_->prefetching);
}
// Tensor 데이터를 사용하여 PartitionBufferStorage 초기화
PartitionBufferStorage::PartitionBufferStorage(string filename, torch::Tensor data, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    options_ = options;
    dtype_ = options_->dtype;
    append(data);
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double) dim0_size_ / options_->num_partitions);
    // 파티션 버퍼 초기화
    buffer_ = new PartitionBuffer(options_->buffer_capacity,
                                  options_->num_partitions,
                                  options_->fine_to_coarse_ratio,
                                  partition_size,
                                  dim1_size_,
                                  dim0_size_,
                                  dtype_,
                                  filename_,
                                  options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    options_ = options;
    dtype_ = options_->dtype;
    int64_t partition_size = ceil((double) dim0_size_ / options_->num_partitions);

    buffer_ = new PartitionBuffer(options_->buffer_capacity,
                                  options_->num_partitions,
                                  options_->fine_to_coarse_ratio,
                                  partition_size,
                                  dim1_size_,
                                  dim0_size_,
                                  dtype_,
                                  filename_,
                                  options_->prefetching);
}
// 파일에 Tensor 값을 특정 위치에 기록 (I/O 병목 가능성 있음)
void PartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);// 파일 열기
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");// 파일 열기 실패 시 예외 발생
    }

    int64_t dtype_size = 0;
    // 데이터 타입에 따른 크기 설정
    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;
    // 파일에 데이터 기록 (I/O 병목 가능성 발생 지점)
    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");// 기록 실패 시 예외 발생
    }

    close(fd);
}
// Tensor 데이터를 파일에 추가
void PartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }
    // 데이터를 파일에 기록 (I/O 작업)
    outfile.write((char *) values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}
// 소멸자: 메모리 해제
PartitionBufferStorage::~PartitionBufferStorage() {
    delete buffer_;
}
// 데이터를 메모리로 로드
void PartitionBufferStorage::load() {
    if (!loaded_ && initialized_) {
        buffer_->load();
        loaded_ = true;
    }
}
// 데이터를 메모리에서 해제
void PartitionBufferStorage::unload(bool write) {
    if (loaded_) {
        buffer_->unload(write);
        loaded_ = false;
    }
}
// 체크포인트 저장: 현재 상태를 파일로 저장
void PartitionBufferStorage::checkpoint(int epoch_id) {
    if (loaded_) {
        buffer_->sync();
    }
    string output_path = filename_.substr(0, filename_.size() - PathConstants::file_ext.size()) + "_" + std::to_string(epoch_id) + PathConstants::file_ext;
    std::ifstream src;
    std::ofstream dst;

    src.open(filename_, ios::in | ios::binary);
    dst.open(output_path, ios::out | ios::binary);

    dst << src.rdbuf();// 파일 복사 (I/O 병목 발생 가능성)

    src.close();
    dst.close();
}
// 인덱스를 사용하여 데이터 읽기 (로드된 데이터로부터)
torch::Tensor PartitionBufferStorage::indexRead(Indices indices) {
    return buffer_->indexRead(indices);
}
// 인덱스를 사용하여 데이터 추가
void PartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) {
    return buffer_->indexAdd(indices, values);
}

torch::Tensor PartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Shuffle not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

void PartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

FlatFile::FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
}

FlatFile::FlatFile(string filename, torch::Tensor data) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    loaded_ = false;
    append(data);
    initialized_ = true;
}

FlatFile::FlatFile(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = dtype;
}
// 지정된 오프셋에 Tensor 데이터를 기록 (I/O 병목 가능성)
void FlatFile::rangePut(int64_t offset, torch::Tensor values) {
    int64_t dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;
    // 파일에 데이터 기록 (I/O 작업)
    if (pwrite_wrapper(fd_, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int64_t dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    outfile.write((char *) values.data_ptr(), values.size(0) * values.size(1) * dtype_size);
    outfile.close();
}


void FlatFile::load() {
    if (!loaded_ && initialized_) {
        fd_ = open(filename_.c_str(), O_RDWR | IO_FLAGS);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        loaded_ = true;
    }
}

void FlatFile::unload(bool write) {
    (void) write;
    if (loaded_) {
        close(fd_);
        loaded_ = false;
    }
}

void FlatFile::checkpoint(int epoch_id) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, use copy instead");
    throw std::runtime_error("");
}

torch::Tensor FlatFile::indexRead(Indices indices) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexAdd(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::move(string new_filename) {
    unload(false);

    renameFile(filename_, new_filename);

    load();
}

void FlatFile::copy(string new_filename, bool rename) {
    unload(false);

    copyFile(filename_, new_filename);

    if (rename) {
        filename_ = new_filename;
    }
    load();
}
// 범위 내 데이터 읽기 (I/O 병목 가능성)
torch::Tensor FlatFile::range(int64_t offset, int64_t n) {
    int dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    torch::Tensor output_tensor = torch::empty({n, dim1_size_}, dtype_);

    if (pread_wrapper(fd_, output_tensor.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    return output_tensor;
}
// 데이터 셔플: 전체 또는 부분적으로 데이터 셔플링
void FlatFile::shuffle() {
    //std::cout << "@@@ In storage.cpp FlatFile shuffle @@@ \n";
    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while(offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SHUFFLE_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SHUFFLE_SIZE;
            }

            torch::Tensor chunk = range(offset, curr_size);
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::randperm(chunk.size(0), opts)));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::sort(bool src) {
    // function for sorting flat file storing edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while(offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SORT_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SORT_SIZE;
            }

            torch::Tensor chunk = range(offset, curr_size);
            // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::argsort(chunk.select(1, sort_dim))));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::mem_load() {
    if (!loaded_) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        SPDLOG_DEBUG("Initialized memory edges");
        process_mem_usage();

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        SPDLOG_DEBUG("Read edges from disk");
        process_mem_usage();

        loaded_ = true;
    }
}

void FlatFile::mem_unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = 0;
        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (write) {
            if (pwrite_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
                SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
                throw std::runtime_error("");
            }
        }

        close(fd_);

        SPDLOG_DEBUG("Edges written");
        process_mem_usage();
        loaded_ = false;
        process_mem_usage();
        data_ = torch::Tensor();
        SPDLOG_DEBUG("Nulled tensor and pointer");
        process_mem_usage();
    }
}

InMemory::InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, torch::DeviceType device) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = device;
}

InMemory::InMemory(string filename, torch::Tensor data, torch::DeviceType device) {
    filename_ = filename;
    dim0_size_ = data.size(0);
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    device_ = device;
    loaded_ = false;

    torch::Tensor temp = data.to(torch::kCPU);

    std::ofstream outfile(filename_, ios::out | ios::binary);

    int64_t dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    outfile.write((char *) temp.data_ptr(), data.size(0) * data.size(1) * dtype_size);

    outfile.close();
}

InMemory::InMemory(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = 0;
    initialized_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
    loaded_ = false;
}

void InMemory::load() {
    if (!loaded_) {
        //auto start = std::chrono::high_resolution_clock::now();  // 시간 측정 시작
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }

        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;
        // 파일에서 데이터를 읽어와 텐서에 저장 (I/O 병목 가능성 발생 지점)
        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
        //auto end = std::chrono::high_resolution_clock::now();  // 시간 측정 종료
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        //std::cout << "In Memory Load Time: " << duration << " ms" << std::endl;

        if (device_ == torch::kCUDA) {
            //auto start_gpu = std::chrono::high_resolution_clock::now();  // GPU 전송 시간 측정 시작
            data_ = data_.to(device_);
            //auto end_gpu = std::chrono::high_resolution_clock::now();  // GPU 전송 시간 측정 종료
            //auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count();
            //std::cout << "In Memory Load CPU to GPU Time: " << duration_gpu << " ms" << std::endl;
        }

        loaded_ = true;
    }
}

void InMemory::force_load() {
    if (!loaded_) {
        load();
    } else {
        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        data_ = data_.to(device_);

        loaded_ = true;
    }
}

void InMemory::unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = 0;

        if (device_ == torch::kCUDA) {
            //auto start_gpu_to_cpu = std::chrono::high_resolution_clock::now();  // GPU에서 CPU로 이동 시간 측정 시작
            data_ = data_.to(torch::kCPU);
            //auto end_gpu_to_cpu = std::chrono::high_resolution_clock::now();  // GPU에서 CPU로 이동 시간 측정 종료
            //auto duration_gpu_to_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_to_cpu - start_gpu_to_cpu).count();
            //std::cout << "GPU to CPU Transfer Time: " << duration_gpu_to_cpu << " ms" << std::endl;
        }

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (write) {
            if (pwrite_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
                SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
                throw std::runtime_error("");
            }
        }

        close(fd_);

        SPDLOG_DEBUG("Written");
        process_mem_usage();

        loaded_ = false;
        data_ = torch::Tensor();
        SPDLOG_DEBUG("Done");
        process_mem_usage();
    }
}

void InMemory::checkpoint(int epoch_id) {
    if (loaded_) {
        unload(true);
    }
    string output_path = filename_.substr(0, filename_.size() - PathConstants::file_ext.size()) + "_" + std::to_string(epoch_id) + PathConstants::file_ext;
    std::ifstream src;
    std::ofstream dst;

    src.open(filename_, ios::in | ios::binary);
    dst.open(output_path, ios::out | ios::binary);

    dst << src.rdbuf();

    src.close();
    dst.close();
}

torch::Tensor InMemory::indexRead(Indices indices) {
    if (data_.defined()) {
        if (data_.device().is_cuda()) {
            return data_.index_select(0, indices.to(device_));
        } else {
            torch::Tensor out;

            if (dtype_ == torch::kFloat32) {
                auto out_options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
                out = torch::empty({indices.size(0), dim1_size_}, out_options);
                torch::index_select_out(out, data_, 0, indices);
            } else if (dtype_ == torch::kInt64) {
                auto out_options = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
                out = torch::empty({indices.size(0), dim1_size_}, out_options);
                torch::index_select_out(out, data_, 0, indices);
            } else if (dtype_ == torch::kInt32) {
                auto out_options = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
                out = torch::empty({indices.size(0), dim1_size_}, out_options);
                torch::index_select_out(out, data_, 0, indices);
            } else {
                SPDLOG_ERROR("Not yet implemented");
                throw std::runtime_error("");
            }

            return out;
        }
    } else {
        return torch::Tensor();
    }

}

void InMemory::indexAdd(Indices indices, torch::Tensor values) {
    if (values.device().is_cuda()) {
        data_.index_add_(0, indices, values);
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
        auto ids_accessor = indices.accessor<int64_t, 1>();
        auto values_accessor = values.accessor<float, 2>();

        int d = values.size(1);
        int64_t size = indices.size(0);
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int j = 0; j < d; j++) {
                data_accessor[ids_accessor[i]][j] += values_accessor[i][j];
            }
        }
    }
}

void InMemory::indexPut(Indices indices, torch::Tensor values) {
    if (values.device().is_cuda()) {
        data_[indices] = values;
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
        auto ids_accessor = indices.accessor<int64_t, 1>();
        auto values_accessor = values.accessor<float, 2>();

        int d = values.size(1);
        int64_t size = indices.size(0);
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int j = 0; j < d; j++) {
                data_accessor[ids_accessor[i]][j] = values_accessor[i][j];
            }
        }
    }
}

torch::Tensor InMemory::range(int64_t offset, int64_t n) {
    return data_.narrow(0, offset, n);
}

void InMemory::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full shuffle
    if (edge_bucket_sizes_.empty()) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::randperm(dim0_size_, opts)));
    }
    // shuffle within edge buckets
    else {
        int64_t start = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void InMemory::sort(bool src) {
    // function for sorting in memory edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full sort
    if (edge_bucket_sizes_.empty()) {
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::argsort(data_.select(1, sort_dim))));
    }
    // sort within edge buckets
    else {
        int64_t start = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}
