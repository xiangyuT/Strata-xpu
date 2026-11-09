#!/usr/bin/env python3
"""Import isolated SYCLomatic 20260928 operator output.

Usage: import-deepseek41.py HOST_OUTPUT KERNEL_OUTPUT
Both inputs are outputs of sycl/tools/migrate.sh with explicit MIGRATE_ONLY.
"""
import pathlib,re,sys

root=pathlib.Path(__file__).resolve().parents[1]
host,kernel=map(pathlib.Path,sys.argv[1:])
# The driver and residency code are maintained SYCL implementations; importing
# raw host migrations over them would silently reintroduce CUDA assumptions.
files={'kernels.dp.cpp':kernel/'src/deepseek41/kernels.dp.cpp'}
for name,source in files.items():
    text=source.read_text()
    # Same explicit/default queue semantics as the maintained port. Restrict
    # these transformations to the newly imported DeepSeek translation units.
    text=re.sub(r'\(dpct::queue_ptr\)\s*((?:\w+(?:->|\.))*\w+)\b',r'strata::q_of(\1)',text)
    text=re.sub(r'static_cast<dpct::queue_ptr>\((\w+)\)',r'strata::q_of(\1)',text)
    if '#include "strata/sycl_queue.hpp"' not in text:
        text=text.replace('#include <dpct/dpct.hpp>','#include <dpct/dpct.hpp>\n#include "strata/sycl_queue.hpp"',1)
    if name=='kernels.dp.cpp':
        start=text.index('__dpct_inline__ uint16_t global_value_bits(')
        end=text.index('\ntemplate <bool Prefetch',start)
        text=text[:start]+'''inline uint16_t global_value_bits(const uint16_t* pointer) {
    // CUDA's ld.global.ca cache policy has no SYCL spelling; the bits are an ordinary F16 load.
    return *pointer;
}
'''+text[end:]
        # Unnamed lambdas avoid SYCLomatic's placeholder template names for
        # CUDA kernels with default template arguments. Work-group barriers in
        # this file require no cooperative root-group launch property.
        text=re.sub(r'parallel_for<\s*dpct_kernel_name<.*?>>\s*\(', 'parallel_for(',text,flags=re.S)
        text=re.sub(r'auto exp_props = sycl::ext::oneapi::experimental::properties\{\s*sycl::ext::oneapi::experimental::use_root_sync\};','',text)
        text=re.sub(r'\bexp_props\s*,\s*','',text)
        text=re.sub(r'__nanosleep\(\d+\);','/* no device nanosleep in SYCL */;',text)
        for header in ['oneapi/dpl/execution','oneapi/dpl/algorithm','dpct/dpl_utils.hpp']:
            text=text.replace('#include <'+header+'>\n','')
        # Use the existing exact stable selector for the optional sorted API.
        # It preserves inputs by copying to its caller-owned scratch. This is
        # a functional fallback, not the CUDA radix sort's performance claim.
        start=text.index('size_t top_k_scratch_bytes(')
        end=text.index('\n}  // namespace strata::deepseek41::cuda',start)
        text=text[:start]+'''size_t top_k_scratch_bytes(int rows) {
    if (rows < 1) throw std::invalid_argument("DeepSeek top-k rows");
    return size_t(rows)*sizeof(float);
}
void top_k_sorted(const float* scores,int32_t* ids,int rows,int k,void* scratch,size_t bytes,void* stream) {
    if (k < 0 || k > rows) throw std::invalid_argument("DeepSeek top-k geometry");
    if (!k) return;
    if (!scratch || bytes < top_k_scratch_bytes(rows)) throw std::invalid_argument("DeepSeek top-k scratch");
    stream_cast(stream)->memcpy(scratch,scores,size_t(rows)*sizeof(float));
    top_k(static_cast<float*>(scratch),ids,rows,k,stream);
}
'''+text[end:]
        # These graph/TP2 waits need a separate coherence port. Fail at the
        # host API instead of launching unbounded GPU polling on mapped data.
        for function in ['wait_host','tp_sum_wait']:
            begin=text.index('void '+function+'(');body=text.index('{',begin)
            depth=1;end=body+1
            while depth:
                depth+=(text[end]=='{')-(text[end]=='}');end+=1
            text=text[:body+1]+'\n    throw std::runtime_error("DeepSeek SYCL graph/TP host waits are not implemented");\n'+text[end-1:]
        text=text.replace('qw += size_t(h)*hidden; kw += size_t(h)*hidden;',
                          'qw += size_t(h)*hidden; if (kw) kw += size_t(h)*hidden;')
        text,count=re.subn(r'dot\s*\+=\s*\(\(r\[i\]\s*\*\s*xn\)\s*\*\s*qw\[i\]\)\s*\*\s*\(\(key\[i\]\s*\*\s*kn\)\s*\*\s*kw\[i\]\);',
                          'dot += kw ? ((r[i]*xn)*qw[i])*((key[i]*kn)*kw[i]) : ((r[i]*xn)*qw[i])*(key[i]*kn);',text)
        if count!=1:raise RuntimeError('unrecognized Engram gate expression; inspect migration before importing')
        # Keep the original migration diagnostics in the isolated raw output.
        text=re.sub(r'/\*\s*DPCT\d+:[\s\S]*?\*/','',text)
    dest=root/'src/deepseek41'/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(text)
    print(dest.relative_to(root))
