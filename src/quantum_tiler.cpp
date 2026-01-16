/**
 * Revolutionary Adaptive Quantum-Inspired Tiling Matrix Multiplication
 * Full C++ Implementation with AVX2/FMA3, OpenMP, rdtsc, and Windows Monitoring
 * 
 * Author: Pedesis TM / SuperGrok Implementation
 * Target: Intel Core i7-7700 (4 cores, 8 threads, AVX2/FMA3)
 * Date: January 2026 - Ashburn/Bull Run, VA
 * 
 * Compile (Visual Studio Developer Command Prompt):
 *   cl /O2 /openmp /arch:AVX2 /EHsc adaptive_simple.cpp pdh.lib
 * 
 * Or with GCC (MinGW/WSL):
 *   g++ -O3 -mavx2 -mfma -fopenmp adaptive_simple.cpp -o adaptive_simple
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <mutex>
#include <numeric>
#include <iomanip>
#include <omp.h>
#include <immintrin.h>  // AVX2/FMA3

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <pdh.h>
#pragma comment(lib, "pdh.lib")
#endif

// ============================================================================
// QUANTUM BARRIER MATHEMATICS
// Based on WKB-style tunneling: V(x) = φ^(-|2x|/δ) - 1
// ============================================================================

const double PHI = (1.0 + std::sqrt(5.0)) / 2.0;
const double LN_PHI = std::log(PHI);
const double ASYMP_PREFACTOR = 1.96;  // 2*sqrt(2)/3 / ln_phi

// Cache specs for i7-7700
const int L1_KB = 32;
const int L2_KB = 256;
const int L3_KB = 8192;
const int L1_LATENCY = 4;
const int L2_LATENCY = 12;
const int L3_LATENCY = 38;
const int DRAM_LATENCY = 200;

double barrier_exponent_asymptotic(double E, double delta = 1.0) {
    if (E >= 0.0) return 0.0;
    if (E <= -1.0) return 1e10;
    return ASYMP_PREFACTOR * delta * std::pow(-E, 1.5);
}

double tunneling_probability(double E, double delta = 1.0) {
    double B = barrier_exponent_asymptotic(E, delta);
    if (B >= 1e9) return 0.0;
    return std::exp(-2.0 * B);
}

double energy_from_latency(double latency_cycles, double max_latency = 200.0) {
    double normalized = std::min(1.0, latency_cycles / max_latency);
    return -normalized;
}

double energy_from_temperature(double temp_celsius, double max_temp = 100.0, double baseline = 40.0) {
    if (temp_celsius <= baseline) return 0.0;
    double normalized = (temp_celsius - baseline) / (max_temp - baseline);
    return -0.3 * normalized;
}

double energy_from_power(double power_watts, double tdp_watts = 65.0) {
    if (power_watts <= tdp_watts * 0.5) return 0.0;
    double normalized = std::min(1.0, (power_watts - tdp_watts * 0.5) / (tdp_watts * 0.5));
    return -0.2 * normalized;
}

int derive_tile_size(double E, double delta = 1.0, int cache_kb = 256, int scale = 32, int min_tile = 32, int max_tile = 128) {
    // min_tile=32 for reasonable performance even under stress
    double B = barrier_exponent_asymptotic(E, delta);
    if (B >= 1e9) return min_tile;
    int tile = static_cast<int>(scale * std::exp(-B) * std::sqrt(static_cast<double>(cache_kb)));
    return std::max(min_tile, std::min(max_tile, tile));
}

double cache_latency_model(int tile_size) {
    int bytes_per_tile = tile_size * tile_size * 4;
    double l1_hit_rate = (bytes_per_tile <= L1_KB * 1024) ? 0.95 : 0.30;
    double l2_hit_rate = 0.35 - (l1_hit_rate - 0.30);
    double l3_hit_rate = 0.25;
    double dram_rate = 1.0 - l1_hit_rate - l2_hit_rate - l3_hit_rate;
    return l1_hit_rate * L1_LATENCY + l2_hit_rate * L2_LATENCY + l3_hit_rate * L3_LATENCY + dram_rate * DRAM_LATENCY;
}

// ============================================================================
// TILE TASK STRUCTURE
// ============================================================================

struct TileTask {
    int i_start, j_start, k_start, tile_size, depth;
    double energy;
    TileTask(int i, int j, int k, int t, int d, double e) 
        : i_start(i), j_start(j), k_start(k), tile_size(t), depth(d), energy(e) {}
};

// ============================================================================
// SYSTEM MONITOR
// ============================================================================

class SystemMonitor {
public:
    struct SystemState {
        double temperature = 50.0;
        double power_watts = 40.0;
        double avg_latency = 20.0;
    };

    SystemMonitor(double poll_interval = 0.001) : poll_interval_(poll_interval), running_(false) {}  // 1ms poll
    
    ~SystemMonitor() { stop(); }

    void start() {
        running_ = true;
        monitor_thread_ = std::thread(&SystemMonitor::monitor_loop, this);
    }

    void stop() {
        running_ = false;
        if (monitor_thread_.joinable()) monitor_thread_.join();
    }

    SystemState get_state() {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_state_;
    }

    void record_latency(double cycles) {
        std::lock_guard<std::mutex> lock(mutex_);
        latency_samples_.push_back(cycles);
        if (latency_samples_.size() > 100) latency_samples_.erase(latency_samples_.begin());
        if (!latency_samples_.empty()) {
            current_state_.avg_latency = std::accumulate(latency_samples_.begin(), latency_samples_.end(), 0.0) / latency_samples_.size();
        }
    }

private:
    double poll_interval_;
    bool running_;
    std::thread monitor_thread_;
    SystemState current_state_;
    std::vector<double> latency_samples_;
    std::mutex mutex_;

    void monitor_loop() {
#ifdef _WIN32
        PDH_HQUERY cpu_query = nullptr;
        PDH_HCOUNTER cpu_counter = nullptr;
        PdhOpenQuery(nullptr, 0, &cpu_query);
        PdhAddEnglishCounterA(cpu_query, "\\Processor(_Total)\\% Processor Time", 0, &cpu_counter);
        PdhCollectQueryData(cpu_query);
#endif
        while (running_) {
            double cpu_percent = 50.0;  // Default
#ifdef _WIN32
            PdhCollectQueryData(cpu_query);
            PDH_FMT_COUNTERVALUE val;
            PdhGetFormattedCounterValue(cpu_counter, PDH_FMT_DOUBLE, nullptr, &val);
            cpu_percent = val.doubleValue;
#endif
            double temp = 40.0 + cpu_percent * 0.5;
            double power = 10.0 + (cpu_percent / 100.0) * 55.0;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                current_state_.temperature = temp;
                current_state_.power_watts = power;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(poll_interval_ * 1000)));
        }
#ifdef _WIN32
        if (cpu_query) PdhCloseQuery(cpu_query);
#endif
    }
};

// ============================================================================
// ADAPTIVE SCHEDULER
// ============================================================================

class AdaptiveScheduler {
public:
    AdaptiveScheduler(int num_threads = 8, int max_depth = 3, double split_threshold = 0.3)
        : num_threads_(num_threads), max_depth_(max_depth), split_threshold_(split_threshold),
          tasks_created_(0), tasks_split_(0) {}

    SystemMonitor monitor;
    int num_threads_, max_depth_;
    double split_threshold_;
    int tasks_created_, tasks_split_;
    std::vector<int> tile_sizes_used_;
    std::vector<double> energies_computed_;

    std::vector<TileTask> create_initial_tasks(int n, int m) {
        monitor.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Quick warm up
        
        auto state = monitor.get_state();
        double delta = std::log(static_cast<double>(n));
        double e_total = std::max(-0.99, energy_from_latency(state.avg_latency) + 
                                  energy_from_temperature(state.temperature) + 
                                  energy_from_power(state.power_watts));
        int tile = derive_tile_size(e_total, delta);
        energies_computed_.push_back(e_total);
        tile_sizes_used_.push_back(tile);

        // Only create (i,j) tasks - k is handled inside to avoid race conditions
        std::vector<TileTask> tasks;
        for (int i = 0; i < n; i += tile) {
            for (int j = 0; j < m; j += tile) {
                tasks.emplace_back(i, j, 0, tile, 0, e_total);  // k=0, will iterate internally
                tasks_created_++;
            }
        }
        return tasks;
    }

    bool should_split(const TileTask& task) {
        if (task.depth >= max_depth_ || task.tile_size <= 16) return false;
        double T = tunneling_probability(task.energy);
        return T < split_threshold_;
    }

    std::vector<TileTask> split_task(const TileTask& task) {
        // Only split over i,j (not k) to avoid race conditions
        // Each sub-task will handle the full k dimension internally
        std::vector<TileTask> sub_tasks;
        int half = task.tile_size / 2;
        auto state = monitor.get_state();
        double e_total = std::max(-0.99, energy_from_latency(state.avg_latency) + 
                                  energy_from_temperature(state.temperature) + 
                                  energy_from_power(state.power_watts));
        for (int di = 0; di < 2; ++di) {
            for (int dj = 0; dj < 2; ++dj) {
                // k_start stays 0 - each sub-task handles full k loop
                sub_tasks.emplace_back(task.i_start + di * half, task.j_start + dj * half, 
                                      0, half, task.depth + 1, e_total);
            }
        }
        tasks_split_++;
        return sub_tasks;
    }
};

// ============================================================================
// AVX2/FMA MATRIX MULTIPLICATION KERNEL
// ============================================================================

inline unsigned long long rdtsc_cycles() {
#ifdef _MSC_VER
    return __rdtsc();
#else
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
#endif
}

// Correct kernel - vectorizes over j (output columns) with broadcast of A[i,k]
// C[i, j:j+8] = Σ_k A[i,k] * B[k, j:j+8]
void matmul_tile_avx2(float* A, float* B, float* C, int i, int j, int k, int tile, int n, int m) {
    int i_end = std::min(i + tile, n);
    int j_end = std::min(j + tile, m);
    int k_end = std::min(k + tile, n);

    for (int ii = i; ii < i_end; ++ii) {
        int jj = j;
        
        // Process 8 output columns at a time (j dimension)
        for (; jj + 8 <= j_end; jj += 8) {
            __m256 sum = _mm256_loadu_ps(&C[ii * m + jj]);  // Load existing C[i, j:j+8]
            
            for (int kk = k; kk < k_end; ++kk) {
                __m256 a_broadcast = _mm256_set1_ps(A[ii * n + kk]);  // Broadcast A[i,k]
                __m256 b_vec = _mm256_loadu_ps(&B[kk * m + jj]);      // Load B[k, j:j+8] - contiguous!
                sum = _mm256_fmadd_ps(a_broadcast, b_vec, sum);
            }
            
            _mm256_storeu_ps(&C[ii * m + jj], sum);  // Store C[i, j:j+8]
        }
        
        // Handle remaining columns (< 8)
        for (; jj < j_end; ++jj) {
            float result = C[ii * m + jj];
            for (int kk = k; kk < k_end; ++kk) {
                result += A[ii * n + kk] * B[kk * m + jj];
            }
            C[ii * m + jj] = result;
        }
    }
}

// Optimized kernel with transposed B (Bt[j,k] = B[k,j])
// For correct matmul: C[i,j] = Σ_k A[i,k] * B[k,j] = Σ_k A[i,k] * Bt[j,k]
// With Bt stored as m×n (j×k), both A[i,k] and Bt[j,k] are contiguous over k
void matmul_tile_avx2_transposed(float* A, float* Bt, float* C, int i, int j, int k, int tile, int n, int m) {
    int i_end = std::min(i + tile, n);
    int j_end = std::min(j + tile, m);
    int k_end = std::min(k + tile, n);

    for (int ii = i; ii < i_end; ++ii) {
        for (int jj = j; jj < j_end; ++jj) {
            __m256 sum = _mm256_setzero_ps();
            int kk = k;
            
            // Both A[i,k:k+8] and Bt[j,k:k+8] are contiguous - perfect for AVX2!
            for (; kk + 8 <= k_end; kk += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[ii * n + kk]);
                __m256 b_vec = _mm256_loadu_ps(&Bt[jj * n + kk]);
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            }
            
            // Horizontal sum
            __m128 lo = _mm256_castps256_ps128(sum);
            __m128 hi = _mm256_extractf128_ps(sum, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            float result = _mm_cvtss_f32(s);
            
            // Handle remainder
            for (; kk < k_end; ++kk) {
                result += A[ii * n + kk] * Bt[jj * n + kk];
            }
            
            C[ii * m + jj] += result;
        }
    }
}

// Transpose B matrix: Bt[j,k] = B[k,j]
void transpose_matrix(float* B, float* Bt, int rows, int cols) {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Bt[j * rows + i] = B[i * cols + j];
        }
    }
}

// ============================================================================
// ADAPTIVE MATMUL
// ============================================================================

void adaptive_matmul(float* A, float* B, float* C, int n, int m, AdaptiveScheduler& scheduler) {
    // Zero result
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n * m; ++i) C[i] = 0.0f;
    
    auto tasks = scheduler.create_initial_tasks(n, m);
    int tile = scheduler.tile_sizes_used_[0];

    int num_tasks = static_cast<int>(tasks.size());
    #pragma omp parallel for num_threads(scheduler.num_threads_) schedule(dynamic)
    for (int idx = 0; idx < num_tasks; ++idx) {
        TileTask task = tasks[idx];
        unsigned long long start = rdtsc_cycles();
        
        // Check if we should split based on tunneling probability
        if (scheduler.should_split(task)) {
            auto sub_tasks = scheduler.split_task(task);
            for (auto& sub : sub_tasks) {
                // Sequential k for each sub-task to avoid races
                for (int sk = 0; sk < n; sk += sub.tile_size) {
                    matmul_tile_avx2(A, B, C, sub.i_start, sub.j_start, sk, sub.tile_size, n, m);
                }
            }
        } else {
            // No split - iterate over k dimension internally
            for (int k = 0; k < n; k += tile) {
                matmul_tile_avx2(A, B, C, task.i_start, task.j_start, k, tile, n, m);
            }
        }
        
        unsigned long long end = rdtsc_cycles();
        scheduler.monitor.record_latency(static_cast<double>(end - start));
    }
    
    scheduler.monitor.stop();
}

// Fast adaptive matmul - uses pre-computed tile size, no runtime monitoring
void adaptive_matmul_fast(float* A, float* B, float* C, int n, int m, int tile, int num_threads) {
    // Zero result
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n * m; ++i) C[i] = 0.0f;
    
    // Create task list
    std::vector<std::pair<int,int>> tasks;
    for (int i = 0; i < n; i += tile) {
        for (int j = 0; j < m; j += tile) {
            tasks.emplace_back(i, j);
        }
    }

    int num_tasks = static_cast<int>(tasks.size());
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int idx = 0; idx < num_tasks; ++idx) {
        int ti = tasks[idx].first;
        int tj = tasks[idx].second;
        
        // Iterate over k dimension internally to avoid race conditions
        for (int k = 0; k < n; k += tile) {
            matmul_tile_avx2(A, B, C, ti, tj, k, tile, n, m);
        }
    }
}

// ============================================================================
// BASELINE FIXED-TILE MATMUL
// ============================================================================

void matmul_baseline(float* A, float* B, float* C, int n, int m, int tile = 64) {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n * m; ++i) C[i] = 0.0f;
    
    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < n; i += tile) {
        for (int j = 0; j < m; j += tile) {
            for (int k = 0; k < n; k += tile) {
                matmul_tile_avx2(A, B, C, i, j, k, tile, n, m);
            }
        }
    }
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

// Adaptive with transposed B - fastest version
void adaptive_matmul_transposed(float* A, float* Bt, float* C, int n, int m, int tile, int num_threads) {
    // Zero result
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n * m; ++i) C[i] = 0.0f;
    
    // Create task list
    std::vector<std::pair<int,int>> tasks;
    for (int i = 0; i < n; i += tile) {
        for (int j = 0; j < m; j += tile) {
            tasks.emplace_back(i, j);
        }
    }

    int num_tasks = static_cast<int>(tasks.size());
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int idx = 0; idx < num_tasks; ++idx) {
        int ti = tasks[idx].first;
        int tj = tasks[idx].second;
        
        for (int k = 0; k < n; k += tile) {
            matmul_tile_avx2_transposed(A, Bt, C, ti, tj, k, tile, n, m);
        }
    }
}

int main(int argc, char* argv[]) {
    int n = 2048, m = 2028, runs = 3;  // Default: 2048x2048 x 2048x2028
    bool stress_mode = false;  // Enable real-time monitoring with splitting
    bool skip_transpose = false;  // Skip transpose benchmark (faster)
    
    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) m = std::atoi(argv[2]);
    if (argc > 3) runs = std::atoi(argv[3]);
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "stress") stress_mode = true;
        if (arg == "notrans") skip_transpose = true;
    }
    
    double total_flops = 2.0 * n * n * m;
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  Quantum-Inspired Adaptive Matrix Multiplication (C++)\n";
    std::cout << "  Intel Core i7-7700 @ 3.60GHz (8 Threads, AVX2, FMA3)\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Allocate
    float* A = new float[n * n];
    float* B = new float[n * m];
    float* Bt = new float[m * n];  // Transposed B
    float* C1 = new float[n * m];
    float* C2 = new float[n * m];
    float* C3 = new float[n * m];
    
    double memory_mb = (n*n + n*m + m*n + 3*n*m) * sizeof(float) / (1024.0 * 1024.0);
    std::cout << "Matrix: " << n << "x" << n << " x " << n << "x" << m << "\n";
    std::cout << "FLOPs: " << std::fixed << std::setprecision(2) << total_flops/1e9 << " GFLOPs\n";
    std::cout << "Memory: " << memory_mb << " MB\n";
    std::cout << "Runs: " << runs << "\n\n";
    
    // Initialize
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n * n; ++i) A[i] = dist(gen);
    for (int i = 0; i < n * m; ++i) B[i] = dist(gen);
    
    // Transpose B once (cost is amortized)
    std::cout << "Transposing B... ";
    auto t_start = std::chrono::high_resolution_clock::now();
    transpose_matrix(B, Bt, n, m);
    auto t_end = std::chrono::high_resolution_clock::now();
    double transpose_time = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << std::fixed << std::setprecision(4) << transpose_time << "s\n\n";
    
    // Baseline benchmark
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Baseline Fixed-Tile (64x64):\n";
    std::cout << std::string(70, '-') << "\n";
    
    double best_baseline = 0;
    for (int r = 0; r < runs; ++r) {
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul_baseline(A, B, C1, n, m, 64);
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        double gflops = total_flops / 1e9 / sec;
        best_baseline = std::max(best_baseline, gflops);
        std::cout << "  Run " << r+1 << ": " << std::setprecision(4) << sec 
                  << "s | " << std::setprecision(2) << gflops << " GFLOPS\n";
    }
    std::cout << "  Best: " << best_baseline << " GFLOPS\n\n";
    
    // Adaptive benchmark (non-transposed)
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Adaptive Quantum-Inspired:\n";
    std::cout << std::string(70, '-') << "\n";
    
    double delta = std::log(static_cast<double>(n));
    double e_default = energy_from_latency(20.0) + energy_from_temperature(50.0) + energy_from_power(40.0);
    int adaptive_tile = derive_tile_size(e_default, delta);
    int num_tasks = (n / adaptive_tile) * (m / adaptive_tile);
    
    std::cout << "  Pre-computed Tile: " << adaptive_tile << " (E=" << std::setprecision(3) << e_default << ")\n";
    std::cout << "  Tasks: " << num_tasks << "\n\n";
    
    double best_adaptive = 0;
    for (int r = 0; r < runs; ++r) {
        auto t1 = std::chrono::high_resolution_clock::now();
        adaptive_matmul_fast(A, B, C2, n, m, adaptive_tile, 8);
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        double gflops = total_flops / 1e9 / sec;
        best_adaptive = std::max(best_adaptive, gflops);
        
        std::cout << "  Run " << r+1 << ": " << std::setprecision(4) << sec 
                  << "s | " << std::setprecision(2) << gflops << " GFLOPS\n";
    }
    std::cout << "  Best: " << best_adaptive << " GFLOPS\n\n";
    
    // Adaptive with B transpose benchmark
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Adaptive + B Transpose (Cache-Optimized):\n";
    std::cout << std::string(70, '-') << "\n";
    
    double best_transposed = 0;
    for (int r = 0; r < runs; ++r) {
        auto t1 = std::chrono::high_resolution_clock::now();
        adaptive_matmul_transposed(A, Bt, C3, n, m, adaptive_tile, 8);
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        double gflops = total_flops / 1e9 / sec;
        best_transposed = std::max(best_transposed, gflops);
        
        std::cout << "  Run " << r+1 << ": " << std::setprecision(4) << sec 
                  << "s | " << std::setprecision(2) << gflops << " GFLOPS\n";
    }
    std::cout << "  Best: " << best_transposed << " GFLOPS\n\n";
    
    // Verify
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Verification:\n";
    double max_err1 = 0.0, max_err2 = 0.0;
    for (int i = 0; i < n * m; ++i) {
        double err1 = static_cast<double>(std::abs(C1[i] - C2[i]));
        double err2 = static_cast<double>(std::abs(C1[i] - C3[i]));
        if (err1 > max_err1) max_err1 = err1;
        if (err2 > max_err2) max_err2 = err2;
    }
    std::cout << "  Adaptive vs Baseline:   " << std::scientific << max_err1 << " " << (max_err1 < 0.01 ? "OK" : "FAIL") << "\n";
    std::cout << "  Transposed vs Baseline: " << std::scientific << max_err2 << " " << (max_err2 < 0.01 ? "OK" : "FAIL") << "\n\n";
    
    // Stress mode: Test with real-time monitoring and splitting
    if (stress_mode) {
        std::cout << std::string(70, '-') << "\n";
        std::cout << "Adaptive with Real-Time Monitoring (Stress Mode):\n";
        std::cout << std::string(70, '-') << "\n";
        
        // Test with lower split threshold to trigger splitting
        double best_stress = 0;
        for (int r = 0; r < runs; ++r) {
            AdaptiveScheduler scheduler(8, 3, 0.3);  // Lower threshold = more splitting
            
            // Zero C2 for reuse
            #pragma omp parallel for num_threads(8)
            for (int i = 0; i < n * m; ++i) C2[i] = 0.0f;
            
            auto t1 = std::chrono::high_resolution_clock::now();
            adaptive_matmul(A, B, C2, n, m, scheduler);
            auto t2 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t2 - t1).count();
            double gflops = total_flops / 1e9 / sec;
            best_stress = std::max(best_stress, gflops);
            
            std::cout << "  Run " << r+1 << ": " << std::setprecision(4) << sec 
                      << "s | " << std::setprecision(2) << gflops << " GFLOPS\n";
            std::cout << "    Tile: " << scheduler.tile_sizes_used_[0]
                      << ", Tasks: " << scheduler.tasks_created_
                      << ", Split: " << scheduler.tasks_split_
                      << ", E=" << std::setprecision(3) << scheduler.energies_computed_[0] << "\n";
        }
        std::cout << "  Best (Stress): " << best_stress << " GFLOPS\n\n";
        
        // Verify stress mode
        double max_err_stress = 0.0;
        for (int i = 0; i < n * m; ++i) {
            double err = static_cast<double>(std::abs(C1[i] - C2[i]));
            if (err > max_err_stress) max_err_stress = err;
        }
        std::cout << "  Stress vs Baseline: " << std::scientific << max_err_stress 
                  << " " << (max_err_stress < 0.01 ? "OK" : "FAIL") << "\n\n";
    }
    
    // Summary
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Summary:\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Baseline (64x64):       " << best_baseline << " GFLOPS\n";
    std::cout << "  Adaptive (128):         " << best_adaptive << " GFLOPS\n";
    std::cout << "  Adaptive + Transpose:   " << best_transposed << " GFLOPS\n";
    std::cout << "  Theoretical Peak:       460.8 GFLOPS (i7-7700 base)\n\n";
    
    double speedup_adaptive = best_adaptive / best_baseline * 100.0;
    std::cout << "  Adaptive vs Baseline:   " << speedup_adaptive << "% (" 
              << (speedup_adaptive > 100 ? "+" : "") << (speedup_adaptive - 100) << "%)\n";
    
    if (stress_mode) {
        std::cout << "\n  Usage: " << argv[0] << " [n] [m] [runs] [stress]\n";
        std::cout << "    stress: Enable real-time monitoring with splitting\n";
    }
    
    delete[] A; delete[] B; delete[] Bt; delete[] C1; delete[] C2; delete[] C3;
    std::cout << "\nDone!\n\n";
    
    return 0;
}
