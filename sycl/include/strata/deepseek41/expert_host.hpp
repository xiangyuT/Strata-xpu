#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <functional>
namespace strata::deepseek41 {
// FP32 cold-expert math, retaining GPU SwiGLU and original quantized bytes.
// Owns bounded input staging and cold-expert scratch; caller uses an in-order queue.
class ExpertHost {
public:
 static bool available();
 ExpertHost(void* queue,int input,int ff,int threads=0,int max_jobs=64,int max_input_rows=1);
 ExpertHost(void* queue,int input,int ff,int threads,int max_jobs,int max_input_rows,bool coefficient_decode);
 ExpertHost(void* queue,int input,int ff,int threads,int max_jobs,int max_input_rows,bool coefficient_decode,bool paired_dots);
 ExpertHost(void* queue,int input,int ff,int threads,int max_jobs,int max_input_rows,bool coefficient_decode,bool paired_dots,bool raw_down_pairs);
 ~ExpertHost();
 void run(int gate_type,int up_type,int down_type,const uint8_t* gate,const uint8_t* up,const uint8_t* down,
          const float* input,float* output,float limit);
 struct Job { const uint8_t *gate,*up,*down; float* output; int input_row=0; };
 void prepare_input(const float* input);
 void prepare_inputs(const float* input,int rows);
 void run_batch(int gate_type,int up_type,int down_type,const Job* jobs,int count,float limit,const std::function<void()>& after_activation={},bool contiguous_rows=false);
 // Synchronous byte packing between math phases, reusing this same CPU pool.
 void copy_expert(uint8_t* destination,const uint8_t* gate,std::size_t gate_bytes,
                  const uint8_t* up,std::size_t up_bytes,const uint8_t* down,std::size_t down_bytes);
 void report() const;
private:
 struct Impl;std::unique_ptr<Impl> p_;
};
}
